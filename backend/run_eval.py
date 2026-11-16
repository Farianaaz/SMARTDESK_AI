"""Evaluation pipeline for SmartDesk AI.

  python -m evals.run_eval                  # run locally, print metric tables, save evals/results.json
  python -m evals.run_eval --no-judge       # skip the LLM-judge metrics (faster, cheaper)
  python -m evals.run_eval --langsmith      # ALSO log datasets + experiments to LangSmith
  python -m evals.run_eval --only routing   # one suite: answerable | unanswerable | routing

Run from the backend/ folder. Needs the KB indexed (python ingest.py) and OPENAI_* in .env.
LangSmith needs LANGSMITH_API_KEY in .env and `pip install -U langsmith` (>= 0.3.13).
"""
import argparse
import json
import logging
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import llm  # noqa: E402
import rag  # noqa: E402
from evals import dataset as ds  # noqa: E402
from evals import evaluators as ev  # noqa: E402

log = logging.getLogger("evals")


# ------------------------------------------------------------------ targets (the code being evaluated)
def rag_target(inputs: dict) -> dict:
    out = rag.answer(inputs["question"])
    return {k: out[k] for k in ("answer", "escalate", "contexts", "cached", "latency_ms")}


def routing_target(inputs: dict) -> dict:
    return {"intent": llm.classify_intent(inputs["message"], strict=True)}


V = ds.DATASET_VERSION
SUITES = [
    {"name": "answerable", "dataset": f"smartdesk-answerable-{V}", "examples": ds.answerable_examples(),
     "target": rag_target,
     "evaluators": [ev.retrieval_hit_at_3, ev.answered_rate, ev.answer_keyword_recall,
                    ev.groundedness, ev.answer_relevance, ev.latency_ms]},
    {"name": "unanswerable", "dataset": f"smartdesk-unanswerable-{V}", "examples": ds.unanswerable_examples(),
     "target": rag_target, "evaluators": [ev.correct_escalation, ev.latency_ms]},
    {"name": "routing", "dataset": f"smartdesk-routing-{V}", "examples": ds.routing_examples(),
     "target": routing_target, "evaluators": [ev.routing_accuracy]},
]


# ------------------------------------------------------------------ local runner
def _percentile(values: list[float], pct: float) -> float:
    values = sorted(values)
    return values[min(len(values) - 1, round(pct / 100 * (len(values) - 1)))]


def run_suite(suite: dict) -> dict:
    rows, errors = [], 0
    for example in suite["examples"]:
        try:
            outputs = suite["target"](example["inputs"])
        except Exception as e:
            errors += 1
            log.warning("target failed on %r: %s", example["inputs"], e)
            continue
        scores = {}
        for evaluator in suite["evaluators"]:
            try:
                result = evaluator(example["inputs"], outputs, example["outputs"])
                scores[result["key"]] = result["score"]
            except Exception as e:
                log.warning("evaluator %s failed: %s", evaluator.__name__, e)
        rows.append({"inputs": example["inputs"], "reference": example["outputs"], "outputs": outputs, "scores": scores})

    metrics = {}
    for key in {k for r in rows for k in r["scores"]}:
        vals = [r["scores"][key] for r in rows if key in r["scores"]]
        if key == "latency_ms":
            metrics["latency_p50_ms"] = round(_percentile(vals, 50), 1)
            metrics["latency_p95_ms"] = round(_percentile(vals, 95), 1)
        else:
            metrics[key] = round(statistics.mean(vals), 3)
    return {"n": len(rows), "errors": errors, "metrics": dict(sorted(metrics.items())), "rows": rows}


def routing_breakdown(rows: list[dict]) -> dict:
    per_intent = {}
    for r in rows:
        expected, got = r["reference"]["intent"], r["outputs"]["intent"]
        per_intent.setdefault(expected, []).append(expected == got)
    return {k: f"{sum(v)}/{len(v)}" for k, v in per_intent.items()}


def cache_test(examples: list[dict]) -> dict:
    """Ask the same questions again: answers should now come from the cache, much faster."""
    cold = [rag_target(e["inputs"]) for e in examples]
    cold_answered = [o for o in cold if not o["escalate"]]
    warm = [rag_target(e["inputs"]) for e in examples]
    warm_answered = [o for o in warm if not o["escalate"]]
    if not cold_answered:
        return {}
    cold_p50 = _percentile([o["latency_ms"] for o in cold_answered], 50)
    warm_p50 = _percentile([o["latency_ms"] for o in warm_answered], 50)
    return {
        "cache_hit_rate_on_repeat": round(sum(o["cached"] for o in warm_answered) / len(warm_answered), 3),
        "latency_p50_ms_first_ask": round(cold_p50, 1),
        "latency_p50_ms_repeat": round(warm_p50, 1),
        "speedup_x": round(cold_p50 / warm_p50, 1) if warm_p50 else None,
    }


def print_table(title: str, metrics: dict, n: int, errors: int):
    print(f"\n### {title}  (n={n}{f', errors={errors}' if errors else ''})")
    print("| Metric | Value |\n|---|---|")
    for k, v in metrics.items():
        print(f"| {k} | {v} |")


# ------------------------------------------------------------------ LangSmith
def run_langsmith(suites: list[dict]):
    if not os.getenv("LANGSMITH_API_KEY"):
        sys.exit("LANGSMITH_API_KEY is not set in .env")
    from langsmith import Client

    client = Client()
    meta = {"model": os.getenv("OPENAI_MODEL"), "rag_threshold": rag.THRESHOLD, "cache_enabled": rag.CACHE_ENABLED}
    for s in suites:
        if not client.has_dataset(dataset_name=s["dataset"]):
            dataset = client.create_dataset(dataset_name=s["dataset"], description=f"SmartDesk AI {s['name']} examples")
            client.create_examples(dataset_id=dataset.id, examples=s["examples"])
            print(f"Created LangSmith dataset {s['dataset']} ({len(s['examples'])} examples)")
        rag.clear_cache()  # every experiment starts with a cold cache
        results = client.evaluate(
            s["target"], data=s["dataset"], evaluators=s["evaluators"],
            experiment_prefix=f"smartdesk-{s['name']}", max_concurrency=1, metadata=meta,
        )
        print(f"LangSmith experiment finished for {s['name']}: {results}")
    print("Open LangSmith > Datasets & Experiments to compare runs.")


# ------------------------------------------------------------------ main
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-judge", action="store_true", help="skip LLM-judge metrics")
    parser.add_argument("--langsmith", action="store_true", help="also log to LangSmith")
    parser.add_argument("--only", choices=[s["name"] for s in SUITES])
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING)

    suites = [s for s in SUITES if not args.only or s["name"] == args.only]
    if args.no_judge:
        for s in suites:
            s["evaluators"] = [e for e in s["evaluators"] if e not in ev.JUDGE_EVALUATORS]

    rag.clear_cache()  # cold start so the numbers are honest
    report = {"timestamp": datetime.now(timezone.utc).isoformat(), "model": os.getenv("OPENAI_MODEL"),
              "rag_threshold": rag.THRESHOLD, "suites": {}}
    started = time.time()
    for s in suites:
        print(f"Running suite: {s['name']} ({len(s['examples'])} examples)...")
        result = run_suite(s)
        report["suites"][s["name"]] = {k: v for k, v in result.items() if k != "rows"}
        report["suites"][s["name"]]["rows"] = result["rows"]
        print_table(s["name"], result["metrics"], result["n"], result["errors"])
        if s["name"] == "routing":
            report["suites"][s["name"]]["per_intent"] = routing_breakdown(result["rows"])
            print("Per-intent correct:", report["suites"][s["name"]]["per_intent"])
        if s["name"] == "answerable" and rag.CACHE_ENABLED:
            report["cache_test"] = cache_test(s["examples"])
            if report["cache_test"]:
                print_table("answer cache (same questions asked twice)", report["cache_test"], len(s["examples"]), 0)

    # headline numbers
    a = report["suites"].get("answerable", {}).get("metrics", {})
    u = report["suites"].get("unanswerable", {}).get("metrics", {})
    if "correct_escalation" in u:
        print(f"\nHallucination rate on unanswerable questions: {round(1 - u['correct_escalation'], 3)}")
    if "answered_rate" in a:
        print(f"False-escalation rate on answerable questions: {round(1 - a['answered_rate'], 3)}")
    print(f"Total time: {time.time() - started:.0f}s")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {out}")

    if args.langsmith:
        run_langsmith(suites)


if __name__ == "__main__":
    main()
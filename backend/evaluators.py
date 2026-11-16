"""Evaluators. Each has the signature LangSmith expects: (inputs, outputs, reference_outputs) -> dict
with a metric name ("key") and a numeric "score". They are plain functions, so the local runner uses them too.

Scores are 0..1 (higher is better) except latency_ms.
"""
import llm


def _fraction(text: str, keywords: list[str]) -> float:
    text = (text or "").lower()
    return sum(k.lower() in text for k in keywords) / len(keywords)


# ------------------------------------------------------------------ answerable suite
def retrieval_hit_at_3(inputs, outputs, reference_outputs):
    """Did the top-3 retrieved KB entries contain every expected fact? (retrieval quality)"""
    found = _fraction("\n".join(outputs["contexts"]), reference_outputs["expected_keywords"])
    return {"key": "retrieval_hit_at_3", "score": 1.0 if found == 1.0 else 0.0}


def answered_rate(inputs, outputs, reference_outputs):
    """The agent answered instead of wrongly escalating. (1 - this = false-escalation rate)"""
    return {"key": "answered_rate", "score": 0.0 if outputs["escalate"] else 1.0}


def answer_keyword_recall(inputs, outputs, reference_outputs):
    """Share of the expected facts that appear in the answer. An escalation scores 0."""
    if outputs["escalate"] or not outputs["answer"]:
        return {"key": "answer_keyword_recall", "score": 0.0}
    return {"key": "answer_keyword_recall",
            "score": _fraction(outputs["answer"], reference_outputs["expected_keywords"])}


GROUND_PROMPT = """You grade a help-desk assistant's ANSWER against the CONTEXT it was given.
Return JSON only: {"score": 1 or 0, "reason": "<short reason>"}.
score = 1 only if EVERY factual claim in the ANSWER (numbers, names, dates, steps, contact details, links)
is stated in the CONTEXT. score = 0 if the ANSWER contains any claim that is not in the CONTEXT."""

RELEVANCE_PROMPT = """You grade a help-desk assistant's ANSWER to a QUESTION.
Return JSON only: {"score": 1 or 0, "reason": "<short reason>"}.
score = 1 if the ANSWER directly addresses what the QUESTION asks. score = 0 if it is off-topic or does not answer it."""


def groundedness(inputs, outputs, reference_outputs):
    """LLM judge: is every claim in the answer supported by the retrieved context? (anti-hallucination)
    An escalation makes no claims, so it scores 1 here and is penalised by answered_rate instead."""
    if outputs["escalate"] or not outputs["answer"]:
        return {"key": "groundedness", "score": 1.0, "comment": "no answer given"}
    verdict = llm.chat_json(GROUND_PROMPT, f"CONTEXT:\n{chr(10).join(outputs['contexts'])}\n\nANSWER:\n{outputs['answer']}")
    return {"key": "groundedness", "score": float(verdict.get("score", 0)), "comment": verdict.get("reason", "")}


def answer_relevance(inputs, outputs, reference_outputs):
    """LLM judge: does the answer address the question?"""
    if outputs["escalate"] or not outputs["answer"]:
        return {"key": "answer_relevance", "score": 0.0, "comment": "no answer given"}
    verdict = llm.chat_json(RELEVANCE_PROMPT, f"QUESTION:\n{inputs['question']}\n\nANSWER:\n{outputs['answer']}")
    return {"key": "answer_relevance", "score": float(verdict.get("score", 0)), "comment": verdict.get("reason", "")}


def latency_ms(inputs, outputs, reference_outputs):
    return {"key": "latency_ms", "score": outputs["latency_ms"]}


# ------------------------------------------------------------------ unanswerable suite
def correct_escalation(inputs, outputs, reference_outputs):
    """Escalated (offered a ticket) instead of answering. (1 - this = hallucination rate)"""
    return {"key": "correct_escalation", "score": 1.0 if outputs["escalate"] else 0.0}


# ------------------------------------------------------------------ routing suite
def routing_accuracy(inputs, outputs, reference_outputs):
    return {"key": "routing_accuracy", "score": 1.0 if outputs["intent"] == reference_outputs["intent"] else 0.0}


JUDGE_EVALUATORS = {groundedness, answer_relevance}  # these cost extra LLM calls; --no-judge skips them
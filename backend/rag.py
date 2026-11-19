"""Hybrid retrieval + grounded answering + escalation decision + semantic answer cache.

RETRIEVAL (RAG_RETRIEVAL in .env):
  dense          semantic vector search only (Chroma)
  hybrid         dense + BM25 keyword search, merged with Reciprocal Rank Fusion
  hybrid_rerank  hybrid, then a cross-encoder re-ranker re-orders the merged candidates   <- default
Why hybrid: dense search matches MEANING ("I can't remember my login" -> password reset) but can miss exact
terms (error codes, product names, extension numbers). BM25 matches exact words but not paraphrases. The
re-ranker reads query + passage together and picks the best few. If rank-bm25 or flashrank is not installed,
the code logs a warning and falls back (hybrid -> dense), so the app keeps working.

CONFIDENCE (when to escalate): answer when EITHER signal is strong, otherwise offer a ticket:
  - the best dense (semantic) cosine distance is <= RAG_DISTANCE_THRESHOLD   (catches paraphrases), or
  - the top re-ranker score is >= RAG_RERANK_ACCEPT                          (catches short or literal questions
    such as "how many sick leaves", whose dense distance is inflated because the query is so short).
After that the LLM must still find the answer in the context, or it replies NOT_ENOUGH_INFO (escalate).
Run `python rag.py` to see both signals for sample queries.

Run:
  python rag.py                      # calibration: prints scores for sample queries and suggests thresholds
  python rag.py "how do I use VPN"   # one query: shows the retrieved hits and scores, asked twice to show the cache

Needs OPENAI_API_KEY and OPENAI_MODEL in .env for the answer step.

CACHE: a repeated (or near-identical) question reuses the earlier LLM answer instead of calling the LLM.
A cached answer is only reused when BOTH hold:
  1. the new question is very close to the cached one (cosine distance <= CACHE_DISTANCE), and
  2. retrieval finds the SAME best KB entry for both.
Rule 2 stops look-alike questions ("sick leave" vs "casual leave") from sharing an answer.
The cache is stored in Chroma, is cleared whenever ingest.py rebuilds the KB, and entries expire after CACHE_TTL_HOURS.
"""
import hashlib
import logging
import os
import re
import sys
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv

from observability import openai_client, traceable

load_dotenv()
log = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION = "smartdesk_kb"
CACHE_COLLECTION = "smartdesk_cache"
TOP_K = 3
# Cosine DISTANCE: lower = closer. Tune this after running the calibration (python rag.py).
THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.58"))

MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")
RETRIEVAL_MODE = os.getenv("RAG_RETRIEVAL", "hybrid_rerank")
CANDIDATES = 10  # per retriever; also the pool size sent to the re-ranker (FlashRank accepts up to 20)
RRF_K = 60       # Reciprocal Rank Fusion constant
BM25_WEIGHT = float(os.getenv("RAG_BM25_WEIGHT", "1.0"))  # <1 trusts dense more, >1 trusts keywords more
BM25_MIN_COVERAGE = float(os.getenv("RAG_BM25_MIN_COVERAGE", "0.5"))  # a keyword hit must contain this share of the query words
RERANK_MIN_SIGNAL = float(os.getenv("RAG_RERANK_MIN_SIGNAL", "0.05"))  # below this the re-ranker is guessing: keep the fused order
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "ms-marco-MiniLM-L-12-v2")
RERANK_ACCEPT = float(os.getenv("RAG_RERANK_ACCEPT") or os.getenv("RAG_RERANK_THRESHOLD") or "0.5")

CACHE_ENABLED = os.getenv("RAG_CACHE", "true").lower() == "true"
CACHE_DISTANCE = float(os.getenv("CACHE_DISTANCE", "0.12"))
CACHE_TTL_SECONDS = float(os.getenv("CACHE_TTL_HOURS", "24")) * 3600
cache_stats = {"hits": 0, "misses": 0}

SYSTEM_PROMPT = (
    "You are the IT and HR help desk assistant for NovaTech Solutions. "
    "Answer ONLY using the context provided. Do not use outside knowledge and do not guess. "
    "Quote numbers, dates, contact details and links exactly as they appear in the context. "
    "Be polite and concise. "
    "If the context does not contain the answer, reply with exactly: NOT_ENOUGH_INFO"
)

_chroma = None
_collection = None
_bm25 = None
_ranker = None
_ranker_failed = False
_warned: set[str] = set()

_STOPWORDS = {"a", "an", "the", "to", "of", "in", "on", "for", "is", "are", "do", "does", "did", "i", "my", "me",
              "we", "you", "your", "how", "what", "can", "it", "and", "or", "with", "at", "be", "by", "from",
              "this", "that", "if", "as", "am", "was", "will", "get", "have", "has", "not", "cannot",
              "don", "doesn", "didn", "isn", "aren", "wasn", "weren", "won", "couldn", "wouldn", "shouldn"}


def _warn_once(message: str) -> None:
    if message not in _warned:
        _warned.add(message)
        log.warning(message)


def _client():
    global _chroma
    if _chroma is None:
        _chroma = chromadb.PersistentClient(path=str(DB_PATH))
    return _chroma


def _get_collection():
    global _collection
    if _collection is None:
        _collection = _client().get_collection(COLLECTION)  # fails if ingest.py was not run
    return _collection


def reset_indexes() -> None:
    """Forget cached handles (used when the KB was re-ingested while the server was running)."""
    global _collection, _bm25
    _collection, _bm25 = None, None


def _tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, minus stopwords, with a light plural stemmer (leaves -> leave)."""
    tokens = []
    text = re.sub(r"['\u2019]\w*", "", text.lower())  # "can't" -> "can", "Solutions'" -> "solutions" (no stray "t" or "s")
    for t in re.findall(r"[a-z0-9]+", text):
        if t in _STOPWORDS or (len(t) == 1 and not t.isdigit()):
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return tokens


def _hit(id_, text, category, distance=None, bm25=None) -> dict:
    return {"id": id_, "text": text, "category": category, "distance": distance, "bm25": bm25,
            "rrf": None, "rerank": None, "dense_best": None}


# ---- dense (semantic) ----
def _dense_search(query: str, n: int) -> list[dict]:
    col = _get_collection()
    res = col.query(query_texts=[query], n_results=min(n, col.count()))
    return [
        _hit(i, d, m["category"], distance=dist)
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# ---- sparse (BM25 keyword) ----
def _bm25_index() -> dict:
    """BM25 index over the same documents that are in Chroma (built once, rebuilt if the KB size changes)."""
    global _bm25
    col = _get_collection()
    count = col.count()
    if _bm25 is None or _bm25["count"] != count:
        from rank_bm25 import BM25Okapi  # pip install rank-bm25

        data = col.get(include=["documents", "metadatas"])
        tokenized = [_tokenize(d) for d in data["documents"]]
        _bm25 = {
            "count": count, "ids": data["ids"], "docs": data["documents"], "metas": data["metadatas"],
            "index": BM25Okapi(tokenized), "tokensets": [set(t) for t in tokenized],
        }
    return _bm25


def _bm25_search(query: str, n: int) -> list[dict]:
    tokens = _tokenize(query)
    if not tokens:
        return []
    idx = _bm25_index()
    scores = idx["index"].get_scores(tokens)
    wanted = set(tokens)

    def coverage(i: int) -> float:  # share of the query's words that appear in this entry
        return len(wanted & idx["tokensets"][i]) / len(wanted)

    # A keyword hit only counts if it contains enough of the query words. Without this, one shared word
    # ("remember", "login") can push an unrelated entry (the VPN guide) above the right one when fused.
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    kept = [i for i in order if scores[i] > 0 and coverage(i) >= BM25_MIN_COVERAGE][:n]
    return [_hit(idx["ids"][i], idx["docs"][i], idx["metas"][i]["category"], bm25=float(scores[i])) for i in kept]


# ---- fusion ----
def _fuse(dense: list[dict], sparse: list[dict]) -> list[dict]:
    """Reciprocal Rank Fusion: an entry ranked high by EITHER retriever (best: both) rises to the top."""
    merged, scores = {}, {}
    for ranked, weight in ((dense, 1.0), (sparse, BM25_WEIGHT)):
        for rank, h in enumerate(ranked, 1):
            scores[h["id"]] = scores.get(h["id"], 0.0) + weight / (RRF_K + rank)
            target = merged.setdefault(h["id"], h)
            if h["distance"] is not None:
                target["distance"] = h["distance"]
            if h["bm25"] is not None:
                target["bm25"] = h["bm25"]
    for id_, h in merged.items():
        h["rrf"] = scores[id_]
    return sorted(merged.values(), key=lambda h: h["rrf"], reverse=True)


# ---- re-ranking ----
def _get_ranker():
    global _ranker, _ranker_failed
    if _ranker is None and not _ranker_failed:
        try:
            from flashrank import Ranker  # pip install flashrank (small ONNX cross-encoder, no PyTorch)

            _ranker = Ranker(model_name=RERANK_MODEL, cache_dir=str(Path(__file__).parent / ".rerank_cache"),
                             max_length=256)
        except Exception as e:
            _ranker_failed = True
            _warn_once(f"Re-ranker unavailable ({e}); using the fused order instead")
    return _ranker


def reranker_available() -> bool:
    return _get_ranker() is not None


def bm25_available() -> bool:
    try:
        import rank_bm25  # noqa: F401
        return True
    except ImportError:
        return False


def _rerank(query: str, candidates: list[dict], k: int) -> list[dict]:
    ranker = _get_ranker()
    if ranker is None or not candidates:
        return candidates[:k]
    from flashrank import RerankRequest

    passages = [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)]
    ranked = ranker.rerank(RerankRequest(query=query, passages=passages))
    scores = {int(r["id"]): float(r["score"]) for r in ranked}
    for i, c in enumerate(candidates):
        c["rerank"] = scores.get(i)
    if max(scores.values(), default=0.0) < RERANK_MIN_SIGNAL:
        # The cross-encoder scores every candidate ~0 for paraphrased questions ("login credentials"): its
        # ordering would be noise, so keep the fused (dense + keyword) order.
        return candidates[:k]
    return [candidates[int(r["id"])] for r in ranked[:k]]


def _retrieve_once(query: str, k: int, mode: str) -> list[dict]:
    dense = _dense_search(query, CANDIDATES)  # always run: its best distance is the calibrated confidence signal
    dense_best = dense[0]["distance"] if dense else None

    if mode != "dense":
        try:
            sparse = _bm25_search(query, CANDIDATES)
        except ImportError:
            _warn_once("rank-bm25 is not installed (pip install rank-bm25); falling back to dense-only retrieval")
            mode, sparse = "dense", []
    if mode == "dense":
        hits = dense[:k]
    elif mode == "bm25":
        hits = sparse[:k]
    else:
        fused = _fuse(dense, sparse)[:CANDIDATES]
        hits = _rerank(query, fused, k) if mode == "hybrid_rerank" else fused[:k]
    for h in hits:
        h["dense_best"] = dense_best
    return hits


def retrieve(query: str, k: int = TOP_K, mode: str | None = None) -> list[dict]:
    """Return the k best KB entries, best first. Each hit has: id, text, category, distance (dense),
    bm25, rrf, rerank (scores that apply to the chosen mode, else None) and dense_best."""
    mode = mode or RETRIEVAL_MODE
    if mode not in MODES:
        raise ValueError(f"Unknown retrieval mode {mode!r}; use one of {MODES}")
    try:
        return _retrieve_once(query, k, mode)
    except Exception:
        reset_indexes()  # the KB may have been re-ingested while the server was running: reconnect once
        return _retrieve_once(query, k, mode)


def confidence(hits: list[dict]) -> tuple[bool, str]:
    """Should the agent answer? Returns (answer?, signal) where signal says which check accepted it."""
    if not hits:
        return False, "none"
    top = hits[0]
    if top.get("dense_best") is not None and top["dense_best"] <= THRESHOLD:
        return True, "dense"
    if top.get("rerank") is not None and top["rerank"] >= RERANK_ACCEPT:
        return True, "rerank"
    return False, "none"


def is_confident(hits: list[dict]) -> bool:
    return confidence(hits)[0]


# ---------------------------------------------------------------- cache
_cache_col = None


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _cache():
    """The Chroma collection that holds cached answers (created once, then the handle is reused)."""
    global _cache_col
    if _cache_col is None:
        try:
            _cache_col = _client().get_collection(CACHE_COLLECTION)
        except Exception:  # first run, or the cache was cleared: create it
            try:
                _cache_col = _client().create_collection(CACHE_COLLECTION, metadata={"hnsw:space": "cosine"})
            except Exception:
                _cache_col = _client().get_or_create_collection(CACHE_COLLECTION)
    return _cache_col


def _with_cache(operation):
    """Run operation(collection). If the handle is stale (for example ingest.py cleared the cache while the
    server was running), reconnect and retry once."""
    global _cache_col
    try:
        return operation(_cache())
    except Exception:
        _cache_col = None
        return operation(_cache())


def _cache_lookup(query: str, top_id: str):
    """Returns (entry_or_None, status) where status is 'hit', 'miss', 'error' or 'disabled'."""
    if not CACHE_ENABLED:
        return None, "disabled"

    def lookup(col):
        if col.count() == 0:
            return None
        res = col.query(query_texts=[_norm(query)], n_results=1)
        if not res["ids"][0]:
            return None
        return res["metadatas"][0][0], res["distances"][0][0]

    try:
        found = _with_cache(lookup)
    except Exception as e:
        log.warning("Cache lookup failed: %s", e)
        cache_stats["misses"] += 1
        return None, "error"
    if found:
        meta, dist = found
        same_entry = meta["top_id"] == top_id
        if dist <= CACHE_DISTANCE and same_entry and time.time() - meta["created"] <= CACHE_TTL_SECONDS:
            cache_stats["hits"] += 1
            return meta, "hit"
        log.info("Cache near-miss: distance=%.3f (max %.2f), same KB entry=%s", dist, CACHE_DISTANCE, same_entry)
    cache_stats["misses"] += 1
    return None, "miss"


def _cache_store(query: str, top_id: str, answer_text: str | None, escalate: bool) -> bool:
    if not CACHE_ENABLED:
        return False

    def store(col):
        col.upsert(
            ids=[hashlib.sha1(_norm(query).encode()).hexdigest()],
            documents=[_norm(query)],
            metadatas=[{"answer": answer_text or "", "escalate": escalate, "top_id": top_id, "created": time.time()}],
        )

    try:
        _with_cache(store)
        return True
    except Exception as e:
        log.warning("Cache store failed: %s", e)
        return False


def clear_cache() -> None:
    global _cache_col
    _cache_col = None
    try:
        _client().delete_collection(CACHE_COLLECTION)
    except Exception:
        pass
    cache_stats.update(hits=0, misses=0)


def get_cache_stats() -> dict:
    total = cache_stats["hits"] + cache_stats["misses"]
    try:
        entries = _with_cache(lambda col: col.count())
    except Exception:
        entries = None
    return {
        "enabled": CACHE_ENABLED,
        "hits": cache_stats["hits"],
        "misses": cache_stats["misses"],
        "hit_rate": round(cache_stats["hits"] / total, 3) if total else None,
        "entries": entries,
    }


# ---------------------------------------------------------------- answering
def _ask_llm(question: str, context: str, retries: int = 2) -> str:
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = openai_client().chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
                ],
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:  # network, rate limit, etc.
            last_error = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after retries: {last_error}")


@traceable(name="rag_answer", run_type="chain")
def answer(query: str) -> dict:
    """Full flow. Returns a dict with:
       answer (str | None), escalate (bool), reason, sources (KB ids), top_distance, contexts (retrieved texts),
       cached (bool: served from the answer cache), cache_status, confidence_signal, latency_ms.

    reason is one of: "answered", "low_confidence" (neither signal was strong enough, no LLM call),
    "llm_not_enough_info" (the LLM found no answer in the context). Both escalation reasons offer a ticket.
    """
    started = time.perf_counter()
    hits = retrieve(query)
    contexts = [h["text"] for h in hits]
    top = hits[0]["dense_best"] if hits else None
    confident, signal = confidence(hits)

    def done(text, escalate, sources, reason, cached=False, cache_status="skipped"):
        latency = round((time.perf_counter() - started) * 1000, 1)
        log.info("RAG %r -> %s | dense=%s rerank=%s accepted_by=%s | cache=%s | %s ms",
                 query, reason, _fmt(top), _fmt(hits[0]["rerank"] if hits else None), signal, cache_status, latency)
        return {
            "answer": text, "escalate": escalate, "reason": reason, "sources": sources, "top_distance": top,
            "contexts": contexts, "cached": cached, "cache_status": cache_status,
            "confidence_signal": signal, "latency_ms": latency,
        }

    if not confident:
        return done(None, True, [], "low_confidence")

    sources = [h["id"] for h in hits]
    entry, status = _cache_lookup(query, hits[0]["id"])
    if entry is not None:
        escalate = bool(entry["escalate"])
        return done(entry["answer"] or None, escalate, [] if escalate else sources,
                    "llm_not_enough_info" if escalate else "answered", cached=True, cache_status="hit")

    reply = _ask_llm(query, "\n\n".join(contexts))
    escalate = "NOT_ENOUGH_INFO" in reply
    text = None if escalate else reply
    _cache_store(query, hits[0]["id"], text, escalate)
    return done(text, escalate, [] if escalate else sources,
                "llm_not_enough_info" if escalate else "answered", cache_status=status)


# ---------------------------------------------------------------- calibration
IN_KB = [
    "how do i reset my password",
    "how many sick leaves do i get",
    "how many sick leaves",          # short phrasing: dense distance is inflated, the re-ranker should accept it
    "sick leave",
    "what time should i come on my first day",
    "when is payday",
    "payday",
    "how do i connect to the vpn",
    "can i work from home",
    "how do i get a software licence",
    "how long is paternity leave",
    "vpn error 809",
    "i can't remember my login credentials",   # paraphrase: the dense distance should accept it
]
NOT_IN_KB = [
    "my monitor is flickering",
    "where can i park my car",
    "my keyboard is not working",
    "projector not working in the conference room",
    "i need an ergonomic chair",
    "how do i book a meeting room",
    "can i bring my dog to the office",
    "what is the weather today",
    "who won the cricket match",
]


def _fmt(x) -> str:
    return "  -  " if x is None else f"{x:.3f}"


def calibrate():
    print(f"Retrieval mode: {RETRIEVAL_MODE}")
    print(f"Gate: ANSWER if best dense distance <= {THRESHOLD}  OR  top re-ranker score >= {RERANK_ACCEPT}\n")
    dense_in, dense_out, rr_out, wrong = [], [], [], []
    for label, queries, should_accept in [
        ("IN the KB  (should be accepted)", IN_KB, True), ("NOT in the KB  (should escalate)", NOT_IN_KB, False)
    ]:
        print(f"=== {label} ===")
        print(f"  {'dense':>6} {'rerank':>7} {'result':>9}  query -> best entry")
        for q in queries:
            hits = retrieve(q)
            top = hits[0]
            ok, signal = confidence(hits)
            (dense_in if should_accept else dense_out).append(top["dense_best"])
            if not should_accept and top["rerank"] is not None:
                rr_out.append(top["rerank"])
            if ok != should_accept:
                wrong.append(q)
            shown = f"{signal}" if ok else "ESCALATE"
            flag = "" if ok == should_accept else "   <-- WRONG"
            print(f"  {_fmt(top['dense_best']):>6} {_fmt(top['rerank']):>7} {shown:>9}  {q!r:46} -> "
                  f"{top['text'].splitlines()[0][:40]}{flag}")
        print()

    total = len(IN_KB) + len(NOT_IN_KB)
    print(f"Correct decisions: {total - len(wrong)}/{total}")
    for q in wrong:
        print(f"  wrong: {q!r}")
    print("\n--- Tuning hints ---")
    print(f"dense distance: in-KB max {max(dense_in):.3f}, out-of-KB min {min(dense_out):.3f}  (RAG_DISTANCE_THRESHOLD={THRESHOLD})")
    if rr_out:
        print(f"re-ranker: out-of-KB max {max(rr_out):.3f}  (RAG_RERANK_ACCEPT={RERANK_ACCEPT}; keep it clearly above this)")
    elif RETRIEVAL_MODE == "hybrid_rerank":
        print("(no re-ranker scores: is flashrank installed? pip install flashrank)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"Retrieval mode: {RETRIEVAL_MODE}\nTop hits:")
        for h in retrieve(question):
            print(f"  {h['id']:8} dense={_fmt(h['distance'])} bm25={_fmt(h['bm25'])} "
                  f"rrf={_fmt(h['rrf'])} rerank={_fmt(h['rerank'])}  {h['text'].splitlines()[0][:55]}")
        print()
        for label in ("1st call", "2nd call (should be a cache hit)"):
            out = answer(question)
            print(f"[{label}] reason={out['reason']} cache={out['cache_status']} latency={out['latency_ms']} ms")
            print("   ", {k: out[k] for k in ("answer", "escalate", "sources")})
        print("\ncache stats:", get_cache_stats())
    else:
        calibrate()
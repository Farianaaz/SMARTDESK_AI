"""Retrieval + grounded answering + escalation decision.

Run:
  python rag.py                      # calibration: prints distances for sample queries
  python rag.py "how do I use VPN"   # one query, full answer flow

Needs OPENAI_API_KEY and OPENAI_MODEL in .env for the answer step.
Retrieval alone needs no API key.
"""
import os
import sys
import time
from pathlib import Path

import chromadb
from dotenv import load_dotenv

load_dotenv()

DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION = "smartdesk_kb"
TOP_K = 3
# Cosine DISTANCE: lower = closer. Tune this after running the calibration (python rag.py).
THRESHOLD = float(os.getenv("RAG_DISTANCE_THRESHOLD", "0.58"))

SYSTEM_PROMPT = (
    "You are the IT and HR help desk assistant for NovaTech Solutions. "
    "Answer ONLY using the context provided. Do not use outside knowledge and do not guess. "
    "Quote numbers, dates, contact details and links exactly as they appear in the context. "
    "Be polite and concise. "
    "If the context does not contain the answer, reply with exactly: NOT_ENOUGH_INFO"
)

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(DB_PATH))
        _collection = client.get_collection(COLLECTION)  # fails if ingest.py was not run
    return _collection


def retrieve(query: str, k: int = TOP_K) -> list[dict]:
    """Return the k closest KB entries as dicts, best match first."""
    res = _get_collection().query(query_texts=[query], n_results=k)
    return [
        {"id": i, "text": d, "category": m["category"], "distance": dist}
        for i, d, m, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        )
    ]


def is_confident(hits: list[dict]) -> bool:
    return bool(hits) and hits[0]["distance"] <= THRESHOLD


def _ask_llm(question: str, context: str, retries: int = 2) -> str:
    from openai import OpenAI  # imported here so retrieval works without the package configured

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    last_error = None
    for attempt in range(retries + 1):
        try:
            resp = client.chat.completions.create(
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


def answer(query: str) -> dict:
    """Full flow. Returns {"answer": str | None, "escalate": bool, "sources": [ids], "top_distance": float}.

    Escalation happens when (1) nothing is retrieved, (2) the best match is too far away,
    or (3) the LLM says the context does not contain the answer.
    """
    hits = retrieve(query)
    top = hits[0]["distance"] if hits else None
    if not is_confident(hits):
        return {"answer": None, "escalate": True, "sources": [], "top_distance": top}

    context = "\n\n".join(h["text"] for h in hits)
    reply = _ask_llm(query, context)
    if "NOT_ENOUGH_INFO" in reply:
        return {"answer": None, "escalate": True, "sources": [], "top_distance": top}
    return {"answer": reply, "escalate": False, "sources": [h["id"] for h in hits], "top_distance": top}


# ---------------- calibration ----------------
IN_KB = [
    "how do i reset my password",
    "how many sick leaves do i get",
    "what time should i come on my first day",
    "when is payday",
    "how do i connect to the vpn",
    "can i work from home",
    "how do i get a software licence",
    "how long is paternity leave",
]
NOT_IN_KB = [
    "my monitor is flickering",
    "where can i park my car",
    "my keyboard is not working",
    "projector not working in the conference room",
    "i need an ergonomic chair",
    "what is the weather today",
    "who won the cricket match",
]


def calibrate():
    for label, queries in [("IN the KB (want SMALL distances)", IN_KB), ("NOT in the KB (want LARGE distances)", NOT_IN_KB)]:
        print(f"\n=== {label} ===")
        dists = []
        for q in queries:
            top = retrieve(q, k=1)[0]
            dists.append(top["distance"])
            first_line = top["text"].splitlines()[0][:60]
            print(f"  {top['distance']:.3f}  {q!r:50} -> {first_line}")
        print(f"  range: {min(dists):.3f} to {max(dists):.3f}")
    print(f"\nCurrent THRESHOLD = {THRESHOLD}. Pick a value between the two ranges "
          f"and set RAG_DISTANCE_THRESHOLD in .env.")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        out = answer(" ".join(sys.argv[1:]))
        print(out)
    else:
        calibrate()
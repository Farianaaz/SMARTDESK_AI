"""Shows that hybrid search is active and what each method returns for the same query.

  python demo_hybrid.py                       # 3 built-in queries (keyword-style and meaning-style)
  python demo_hybrid.py "your own question"   # any question
"""
import sys

import rag

DEFAULT_QUERIES = [
    "I can't remember my login credentials",    # meaning, few shared words -> dense should find it
    "vpn error 809",                            # exact term -> BM25 should find it
    "how many days off can I take when ill",    # meaning -> dense / re-ranker
]


def first_line(text: str) -> str:
    return text.splitlines()[0][:70]


def main():
    queries = sys.argv[1:] or DEFAULT_QUERIES
    print("Active configuration")
    print(f"  RAG_RETRIEVAL         = {rag.RETRIEVAL_MODE}")
    print("  dense (Chroma)        = OK")
    print(f"  BM25 (rank-bm25)      = {'OK' if rag.bm25_available() else 'MISSING -> pip install rank-bm25'}")
    print(f"  re-ranker (flashrank) = {'OK' if rag.reranker_available() else 'MISSING -> pip install flashrank'}")
    if rag.RETRIEVAL_MODE == "dense":
        print("  NOTE: RAG_RETRIEVAL=dense in your .env, so only semantic search is used. "
              "Set it to hybrid_rerank.")

    labels = {"dense": "DENSE only (semantic)", "bm25": "BM25 only (keywords)",
              "hybrid": "HYBRID (dense + BM25, fused)", "hybrid_rerank": "HYBRID + RE-RANKER (what the agent uses)"}
    for query in queries:
        print(f"\n=== {query!r}")
        for mode in rag.MODES:
            hits = rag.retrieve(query, k=3, mode=mode)
            print(f"  {labels[mode]}")
            for h in hits:
                scores = []
                if h["distance"] is not None:
                    scores.append(f"dense {h['distance']:.2f}")
                if h["bm25"] is not None:
                    scores.append(f"bm25 {h['bm25']:.1f}")
                if h["rerank"] is not None:
                    scores.append(f"rerank {h['rerank']:.2f}")
                print(f"    {h['id']:8} {first_line(h['text']):70}  [{', '.join(scores)}]")


if __name__ == "__main__":
    main()
"""Build the vector index from every JSON file in kb/.

Run:  python ingest.py
Re-run it any time you edit the KB; the collection is rebuilt from scratch.
Each Q&A pair is stored as ONE chunk (answers are short and self-contained).
"""
import json
from collections import Counter
from pathlib import Path

import chromadb

BASE = Path(__file__).parent
KB_DIR = BASE / "kb"
DB_PATH = BASE / "chroma_db"
COLLECTION = "smartdesk_kb"
CACHE_COLLECTION = "smartdesk_cache"


def load_entries() -> list[dict]:
    entries = []
    for f in sorted(KB_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        print(f"  {f.name}: {len(data)} entries")
        entries.extend(data)
    return entries


def main():
    print("Loading KB files...")
    entries = load_entries()

    dup_ids = [i for i, n in Counter(e["id"] for e in entries).items() if n > 1]
    if dup_ids:
        raise SystemExit(f"Duplicate ids found, fix before indexing: {dup_ids[:5]}")

    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass  # first run: nothing to delete
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    try:
        client.delete_collection(CACHE_COLLECTION)  # cached answers may be stale after a KB change
        print("Cleared the answer cache")
    except Exception:
        pass

    col.add(
        ids=[e["id"] for e in entries],
        documents=[f"Q: {e['question']}\nA: {e['answer']}" for e in entries],
        metadatas=[{"category": e["category"]} for e in entries],
    )
    print(f"\nIndexed {col.count()} entries into {DB_PATH.name}/")


if __name__ == "__main__":
    main()
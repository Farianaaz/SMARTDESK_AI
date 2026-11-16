import json
import re
from pathlib import Path
from datasets import load_dataset

DATASET = "strova-ai/hr-policies-qa-dataset"
MAX_ENTRIES = 300

ds = load_dataset(DATASET, split="train")


def clean(text):
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"([a-z])(laws|policies)\b", r"\1 \2", text) 


seen, entries = set(), []
for row in ds:
    q = None
    for m in row["messages"]:
        if m["role"] == "user":
            q = clean(m["content"])
        elif m["role"] == "assistant" and q:
            a = clean(m["content"])
            key = q.lower()
            if key not in seen and len(a) >= 20:
                seen.add(key)
                entries.append({
                    "id": f"hr-{len(entries) + 1:03d}",
                    "category": "HR",
                    "question": q,
                    "answer": a,
                })
            q = None
    if len(entries) >= MAX_ENTRIES:
        break

Path("kb").mkdir(exist_ok=True)
Path("kb/hr_policies_kb.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote {len(entries)} entries to kb/kb.json")
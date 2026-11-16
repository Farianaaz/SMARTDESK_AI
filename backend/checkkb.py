import json
from collections import Counter
from pathlib import Path

entries = []
for f in sorted(Path("kb").glob("*.json")):
    data = json.loads(f.read_text(encoding="utf-8"))
    print(f"{f.name:24} {len(data)} entries")
    entries += data

text = lambda e: (e["question"] + " " + e["answer"]).lower()
problems = []

print("\nTotal:", len(entries))
print("By category:", dict(Counter(e["category"] for e in entries)))

# 1. unique ids
dup_ids = [i for i, n in Counter(e["id"] for e in entries).items() if n > 1]
if dup_ids:
    problems.append(f"Duplicate ids: {dup_ids[:5]}")

# 2. duplicate questions
dup_q = [q for q, n in Counter(e["question"].strip().lower() for e in entries).items() if n > 1]
if dup_q:
    problems.append(f"{len(dup_q)} duplicate questions, e.g. {dup_q[:3]}")

# 3. leftovers from the other company
if any("kreeda" in text(e) for e in entries):
    problems.append("'Kreeda' still present")

# 4. the raw HF file must not be in kb/
if Path("kb/kb.json").exists():
    problems.append("kb/kb.json is still in kb/ (run curate_kb.py)")

# 5. required topics must be covered
print("\nTopic coverage (0 = missing):")
for kw in ["password", "vpn", "mfa", "wi-fi", "software", "sick leave", "casual leave", "earned leave",
           "maternity", "work from home", "reimburs", "first day", "laptop", "insurance", "probation",
           "payday", "tax", "vesting", "notice period", "code of conduct", "harass", "performance"]:
    n = sum(kw in text(e) for e in entries)
    print(f"  {kw:16} {n}")
    if n == 0:
        problems.append(f"No coverage for '{kw}'")

# 6. deliberate gaps must stay OUT of the KB
import re
gap_patterns = {
    "monitor (hardware)": r"\bmonitors?\b.*\b(screen|display|flicker|hdmi|resolution)\b|\b(screen|display|flicker)\w*\b.*\bmonitors?\b",
    "flicker": r"flicker",
    "parking": r"\bparking\b",
    "keyboard": r"\bkeyboards?\b",
    "mouse": r"\bmouse\b",
    "docking": r"docking station",
    "projector": r"projector",
}
for name, pat in gap_patterns.items():
    hits = [e["id"] for e in entries if re.search(pat, text(e))]
    if hits:
        problems.append(f"Gap topic '{name}' appears in {hits[:3]}; escalation tests will break")

# 7. conflicting numbers
for kw, expected in [("compensatory", "30 days"), ("sick leave", "8 days")]:
    for e in entries:
        t = text(e)
        if kw in t and "60 days" in t:
            problems.append(f"{e['id']} conflicts on {kw}")

print("\n" + ("READY" if not problems and len(entries) >= 30 else "NOT READY"))
for p in problems:
    print(" -", p)
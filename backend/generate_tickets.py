import json
import os
import re
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL")

# Topics intentionally NOT in the knowledge base
TOPICS = [
    "monitor flickering or no display",
    "office parking",
    "keyboard or mouse not working",
    "docking station problems",
    "projector or conference room equipment",
    "badge or door access not working",
    "headset or webcam issues",
    "ergonomic chair or desk request",
]
TICKETS_PER_TOPIC = 3
VALID_PRIORITIES = {"Low", "Medium", "High", "Critical"}

PROMPT = """You are generating realistic internal help-desk tickets for NovaTech Solutions,
a 500-employee SaaS company in Bengaluru.

Write {n} different tickets an employee might raise about: {topic}.
Vary the tone (some terse, some detailed, some frustrated) and the urgency.

Output ONLY a JSON array, with no other text, in exactly this shape:
[{{"summary": "one-line title", "description": "2-4 sentence description written by the employee",
   "category": "IT", "priority": "Low|Medium|High|Critical"}}]
"""


def parse_json_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array in response")
    return json.loads(text[start : end + 1])


def generate(topic: str) -> list[dict]:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": PROMPT.format(n=TICKETS_PER_TOPIC, topic=topic)}],
    )
    tickets = parse_json_array(resp.choices[0].message.content)
    cleaned = []
    for t in tickets:
        if not all(k in t for k in ("summary", "description", "category", "priority")):
            continue
        if t["priority"] not in VALID_PRIORITIES:
            t["priority"] = "Medium"
        t["topic"] = topic
        t["description"] = re.sub(r"\s+", " ", t["description"]).strip()
        cleaned.append(t)
    return cleaned


if __name__ == "__main__":
    all_tickets = []
    for topic in TOPICS:
        try:
            batch = generate(topic)
            all_tickets.extend(batch)
            print(f"{topic}: {len(batch)} tickets")
        except Exception as e:
            print(f"{topic}: failed ({e})")

    Path("tests").mkdir(exist_ok=True)
    out = Path("tests/generated_tickets.json")
    out.write_text(json.dumps(all_tickets, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved {len(all_tickets)} tickets to {out}")
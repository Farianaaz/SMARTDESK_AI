import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = os.getenv("OPENAI_MODEL")

FACTS = """
Company: NovaTech Solutions, 500-employee SaaS company, Bengaluru, India
Leave: casual leave 12 days/year, sick leave 8 days/year (medical certificate needed for more than 2 consecutive days),
  earned leave 15 days/year (up to 30 days can be carried forward), maternity leave 26 weeks, paternity leave 2 weeks,
  compensatory off for approved weekend work (use within 30 days, otherwise it lapses). Leave is applied for in the HR portal and approved by the manager.
  Unused casual leave and sick leave lapse at year end and cannot be carried forward.
  Casual leave can be taken in half-day units. Employees get a HR portal notification when leave is approved.
WFH: up to 2 days per week with manager approval. Core hours 11 AM to 4 PM IST. Employees must be reachable on Teams.
Payday: last working day of each month. Payslips are in the HR portal. Salary is credited to the registered bank account.
  Bank account changes are made in the HR portal. HR portal access ends on the last working day.
Tax: investment declarations are submitted in the HR portal between April and June, with proofs by January. Proofs submitted after January are not accepted.
Reimbursement: submit receipts within 30 days through the expense portal. Approved claims are paid with the next payroll.
Stock options: 4-year vesting with a 1-year cliff, then monthly vesting.
Probation: 6 months, with a review at 3 months and a final review at 6 months.
Notice period: 60 days after confirmation. Final settlement includes unpaid salary, approved reimbursements and encashment of unused earned leave.
Onboarding: day one starts at 9:30 AM at reception with an HR induction. Laptop and ID badge are issued on day one.
  A buddy is assigned for the first 3 months. Insurance enrolment must be completed within 30 days of joining.
  Health insurance covers the employee, spouse and up to 2 children.
IT Helpdesk: ext. 2020, helpdesk@novatech.example. HR Helpdesk: hr@novatech.example, ext. 2010.
"""

# (output file, id prefix, category, topic, subtopics to cover)
JOBS = [
    ("onboarding_kb.json", "onb", "HR", "Employee onboarding",
     "day-one checklist, laptop and equipment allocation, ID badge, benefits and insurance enrolment, "
     "buddy programme, probation period and reviews, first-week schedule, documents to bring"),
    ("payroll_kb.json", "pay", "HR", "Payroll and compensation",
     "payday schedule, payslips, tax declaration, salary structure basics, stock-option vesting, "
     "reimbursement process, salary credit issues, notice period and final settlement"),
    ("leave_wfh_kb.json", "lv", "HR", "Leave and work-from-home policy",
     "casual, sick and earned leave, maternity and paternity leave, carry-forward, compensatory off, "
     "how to apply for leave, WFH rules, core hours, approval process"),
]
PER_JOB = 20

PROMPT = """You are an HR manager at the company described in the facts below.
Use ONLY these facts. Every number in your answers must match them exactly.
If a detail is not in the facts, keep the answer general or direct the employee to HR. Never invent specific numbers, dates or URLs.

FACTS:
{facts}

Generate {n} realistic employee questions and answers about: {topic}.
Cover these subtopics (several questions each): {subtopics}.
Mix simple factual questions, step-by-step procedural questions, and policy-interpretation questions
(for example, "Can I carry forward unused sick leave?"). Answers should be 2 to 4 sentences, professional and friendly.

Output ONLY a JSON array, with no other text:
[{{"question": "...", "answer": "..."}}]
"""


def parse_json_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("No JSON array in response")
    return json.loads(text[start : end + 1])


def generate(prefix, category, topic, subtopics):
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=4000,
        temperature=0.4,
        messages=[{"role": "user", "content": PROMPT.format(
            facts=FACTS, n=PER_JOB, topic=topic, subtopics=subtopics)}],
    )
    items = parse_json_array(resp.choices[0].message.content)
    entries, seen = [], set()
    for it in items:
        q = re.sub(r"\s+", " ", str(it.get("question", ""))).strip()
        a = re.sub(r"\s+", " ", str(it.get("answer", ""))).strip()
        if not q or len(a) < 20 or q.lower() in seen:
            continue
        seen.add(q.lower())
        entries.append({"id": f"{prefix}-{len(entries) + 1:03d}", "category": category,
                        "question": q, "answer": a})
    return entries


if __name__ == "__main__":
    Path("kb").mkdir(exist_ok=True)
    for filename, prefix, category, topic, subtopics in JOBS:
        try:
            entries = generate(prefix, category, topic, subtopics)
            Path("kb", filename).write_text(
                json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{filename}: {len(entries)} entries")
        except Exception as e:
            print(f"{filename}: failed ({e})")
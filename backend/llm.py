"""Small LLM helpers used by the graph: intent classification and ticket field inference.
Both fall back to simple keyword rules if the LLM call fails, so the agent keeps working.
Intent results are cached in memory, so a repeated message costs no LLM call."""
import json
import logging
import os
import re
import time

from dotenv import load_dotenv

from observability import openai_client

load_dotenv()
log = logging.getLogger(__name__)

INTENTS = {"kb_query", "create_ticket", "ticket_status", "manage_email", "smalltalk", "out_of_scope"}

INTENT_PROMPT = """You route messages for the IT and HR help desk of NovaTech Solutions.
Reply with JSON only: {"intent": "<label>"} where <label> is exactly one of:
- kb_query: an IT or HR question, OR a description of a problem the employee is having (try the knowledge base first).
- create_ticket: the employee explicitly asks to raise, create, open or log a support ticket.
- ticket_status: the employee asks about the status or updates of a ticket they raised earlier.
- manage_email: the employee asks SmartDesk to forget, clear, change or update the email address SmartDesk has saved for them. Questions about using email software are NOT this.
- smalltalk: greetings, thanks, goodbyes.
- out_of_scope: anything unrelated to IT or HR at work (weather, sports, general knowledge, coding help, etc.).
Use create_ticket ONLY when a ticket is explicitly requested."""

FIELDS_PROMPT = """You prepare a help-desk ticket from an employee's issue description.
Reply with JSON only: {"summary": "<one-line title, max 80 chars>", "category": "IT" or "HR",
"priority": "Low" | "Medium" | "High" | "Critical"}.
Priority: Critical = work completely blocked or security incident; High = work significantly affected;
Medium = normal; Low = minor or a request."""

HR_WORDS = ("leave", "salary", "payroll", "payslip", "insurance", "wfh", "reimburse", "tax", "vesting",
            "probation", "onboarding", "bonus", "appraisal", "holiday", "policy")
GREETINGS = ("hi", "hello", "hey", "thanks", "thank you", "bye", "good morning", "good evening")
_FORGET_EMAIL = re.compile(r"\b(forget|clear|remove|delete)\b.{0,15}\b(saved )?email\b|\b(change|update) my saved email\b|\bwrong email\b")

_intent_cache: dict[str, str] = {}


def _chat_json(system: str, user: str, retries: int = 1) -> dict:
    last = None
    for attempt in range(retries + 1):
        try:
            resp = openai_client().chat.completions.create(
                model=os.getenv("OPENAI_MODEL"),
                temperature=0,
                response_format={"type": "json_object"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            last = e
            time.sleep(1 + attempt)
    raise RuntimeError(f"LLM call failed: {last}")


chat_json = _chat_json  # public name, also used by the evaluation judges


def _keyword_intent(message: str) -> str:
    m = message.lower().strip(" !.?")
    if m in GREETINGS:
        return "smalltalk"
    if _FORGET_EMAIL.search(m):
        return "manage_email"
    if any(k in m for k in ("status", "any update", "updates on", "my ticket")):
        return "ticket_status"
    if any(k in m for k in ("raise a ticket", "create a ticket", "open a ticket", "log a ticket")):
        return "create_ticket"
    return "kb_query"


def classify_intent(message: str, strict: bool = False) -> str:
    """strict=True raises instead of falling back to keywords (used by the evaluation)."""
    key = " ".join(message.lower().split())
    if key in _intent_cache:
        return _intent_cache[key]
    try:
        intent = _chat_json(INTENT_PROMPT, message).get("intent")
        if intent in INTENTS:
            if len(_intent_cache) >= 1000:
                _intent_cache.clear()
            _intent_cache[key] = intent
            return intent
        raise ValueError(f"Unexpected intent {intent!r}")
    except Exception as e:
        if strict:
            raise
        log.warning("Intent LLM failed (%s), using keyword fallback", e)
    return _keyword_intent(message)


def infer_ticket_fields(issue: str) -> dict:
    fallback = {
        "summary": issue.strip()[:80],
        "category": "HR" if any(w in issue.lower() for w in HR_WORDS) else "IT",
        "priority": "Medium",
    }
    try:
        out = _chat_json(FIELDS_PROMPT, issue)
        return {
            "summary": str(out.get("summary") or fallback["summary"])[:80],
            "category": out.get("category") if out.get("category") in ("IT", "HR") else fallback["category"],
            "priority": out.get("priority") if out.get("priority") in ("Low", "Medium", "High", "Critical") else "Medium",
        }
    except Exception as e:
        log.warning("Ticket-fields LLM failed (%s), using fallback", e)
        return fallback
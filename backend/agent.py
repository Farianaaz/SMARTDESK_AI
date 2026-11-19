"""LangGraph orchestration for SmartDesk AI.

Every user message runs through `classify`, which routes to a branch:

  kb_query      -> rag_answer -> (not confident) -> offer_ticket -> draft_ticket ...
  create_ticket -> draft_ticket -> ensure_email -> confirm_ticket -> submit_ticket
  ticket_status -> ensure_email -> ticket_status
  manage_email  -> manage_email (forget / change the saved email)
  smalltalk / out_of_scope -> canned reply

Human-in-the-loop uses LangGraph `interrupt()`: when a node needs more input (email,
yes/no confirmation, which ticket to show) the graph pauses, the question goes to the user,
and their next message resumes the graph exactly where it stopped. State (including the
email, so it is never asked twice) is kept per session by a checkpointer.

Run:  python agent.py             # chat in the terminal
      python agent.py --diagram   # print a Mermaid diagram for your README
"""
import logging
import re
import sys
import uuid
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

import llm
import profiles
import rag
import tickets

log = logging.getLogger(__name__)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
YES = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead", "confirm", "do it"}
NO = {"no", "n", "nope", "nah"}
CANCEL = {"cancel", "stop", "never mind", "nevermind", "forget it", "quit"}


class AgentState(TypedDict, total=False):
    message: str                 # latest user message
    intent: str
    flow: str | None             # "create" or "status": decides what follows ensure_email
    escalate: bool               # RAG could not answer
    cached: bool                 # the answer came from the answer cache
    wants_ticket: bool
    reroute: bool                # user typed something new instead of yes/no
    cancelled: bool
    confirmed: bool
    user_id: str | None          # anonymous browser id: lets the saved email follow the user across sessions
    email: str | None            # saved per user (profiles.py), so it is asked once
    last_question: str | None    # the unanswered question, used to draft the ticket
    draft: dict
    reply: str


# ---------------------------------------------------------------- helpers
def _norm(text: str) -> str:
    return text.lower().strip(" .!?")


def _is_yes(text: str) -> bool:
    n = _norm(text)
    return n in YES or n.startswith("yes")


def _is_no(text: str) -> bool:
    return _norm(text) in NO


def _is_cancel(text: str) -> bool:
    return _norm(text) in CANCEL


def _ask(prompt: str) -> str:
    """Pause the graph and wait for the user's next message."""
    return str(interrupt(prompt)).strip()


def _ticket_summary(draft: dict, email: str) -> str:
    return (
        "Here is the ticket I will create:\n"
        f"- Title: {draft['summary']}\n"
        f"- Description: {draft['description']}\n"
        f"- Category: {draft['category']}\n"
        f"- Priority: {draft['priority']}\n"
        f"- Email: {email}\n\n"
        "Shall I go ahead? (yes/no) You can also type extra details to add them. "
        "(Not your email? Say 'change my email'.)"
    )


# ---------------------------------------------------------------- nodes
def classify(state: AgentState):
    intent = llm.classify_intent(state["message"])
    return {  # reset all per-turn fields; email persists
        "intent": intent,
        "flow": "status" if intent == "ticket_status" else None,
        "escalate": False,
        "cached": False,
        "wants_ticket": False,
        "reroute": False,
        "cancelled": False,
        "confirmed": False,
        "last_question": None,
        "reply": "",
    }


def rag_answer(state: AgentState):
    question = state["message"]
    try:
        result = rag.answer(question)
    except Exception:
        log.exception("RAG failed")
        return {"reply": "I'm having trouble reaching my knowledge base right now. Please try again in a few minutes."}
    if result["escalate"]:
        return {"escalate": True, "last_question": question}
    return {"reply": result["answer"], "cached": result["cached"]}


def offer_ticket(state: AgentState):
    ans = _ask("I don't have enough information to answer that. Would you like me to create a support ticket? (yes/no)")
    if _is_yes(ans):
        return {"wants_ticket": True}
    if _is_no(ans) or _is_cancel(ans):
        return {"reply": "No problem. Anything else I can help with?"}
    return {"message": ans, "reroute": True}  # a new question: send it back through classify


def draft_ticket(state: AgentState):
    issue = state.get("last_question")
    if not issue:
        issue = _ask("Sure. Please describe the issue in a sentence or two.")
        if _is_cancel(issue):
            return {"cancelled": True, "reply": "Okay, I've cancelled that. Anything else I can help with?"}
    fields = llm.infer_ticket_fields(issue)
    return {
        "flow": "create",
        "last_question": issue,
        "draft": {
            "summary": fields["summary"],
            "description": f"Employee reports: {issue}",
            "category": fields["category"],
            "priority": fields["priority"],
        },
    }


def manage_email(state: AgentState):
    """Let the user forget or change the email SmartDesk has saved for them."""
    msg = state["message"]
    user_id = state.get("user_id")
    if re.search(r"\b(forget|clear|remove|delete)\b", msg, re.I):
        profiles.forget(user_id)
        return {"email": None, "reply": "Done. I've forgotten your saved email address. I'll ask for it again next time I need it."}
    match = EMAIL_RE.search(msg)
    if not match:
        ans = _ask("Sure. Which email address should I use from now on?")
        match = None if _is_cancel(ans) else EMAIL_RE.search(ans)
        if not match:
            return {"reply": "Okay, I've kept your current email address."}
    email = match.group(0).lower()
    profiles.set_email(user_id, email)
    return {"email": email, "reply": f"Got it. I'll use {email} from now on."}


def ensure_email(state: AgentState):
    if state.get("email"):
        return {"email": state["email"]}
    prompt = (
        "Could you share your work email address so I can link the ticket to you?"
        if state.get("flow") == "create"
        else "Sure. What is your work email address?"
    )
    while True:
        ans = _ask(prompt)
        if _is_cancel(ans):
            return {"cancelled": True, "reply": "Okay, I've cancelled that. Anything else I can help with?"}
        match = EMAIL_RE.search(ans)
        if match:
            email = match.group(0).lower()
            profiles.set_email(state.get("user_id"), email)  # remember it for future sessions
            return {"email": email}
        prompt = "That doesn't look like a valid email address. Please try again, or type 'cancel'."


def confirm_ticket(state: AgentState):
    draft = dict(state["draft"])
    prompt = _ticket_summary(draft, state["email"])
    while True:
        ans = _ask(prompt)
        if _is_yes(ans):
            return {"confirmed": True, "draft": draft}
        if _is_no(ans) or _is_cancel(ans):
            return {"confirmed": False, "reply": "Okay, I won't create the ticket. Anything else I can help with?"}
        draft["description"] += f" Additional details: {ans}"  # employee added more detail
        prompt = "I've added that. " + _ticket_summary(draft, state["email"])


def submit_ticket(state: AgentState):
    d = state["draft"]
    try:
        t = tickets.create_ticket(
            email=state["email"], summary=d["summary"], description=d["description"],
            category=d["category"], priority=d["priority"],
        )
    except Exception:
        log.exception("Ticket creation failed")
        return {"reply": "I couldn't reach the ticketing system, so the ticket was not created. Please try again in a few minutes."}
    return {
        "reply": f"Done! Ticket {t['id']} has been created. Track it here: {t['url']} or ask me for its status anytime.",
        "draft": {},
        "last_question": None,
    }


def _pick_ticket(found: list[dict], answer: str):
    a = answer.strip().lower()
    if a.isdigit() and 1 <= int(a) <= len(found):
        return found[int(a) - 1]
    for t in found:
        if t["id"].lower() in a:
            return t
    return None


def _ticket_details(t: dict) -> str:
    update = t["comments"][-1] if t.get("comments") else "No updates from the support team yet."
    return (
        f"Ticket {t['id']}: {t['summary']}\n"
        f"Status: {t['status']}\nPriority: {t['priority']}\nLatest update: {update}"
    )


def ticket_status(state: AgentState):
    email = state["email"]
    try:
        found = tickets.get_tickets(email)
    except Exception:
        log.exception("Ticket lookup failed")
        return {"reply": "I couldn't reach the ticketing system right now. Please try again in a few minutes."}
    if not found:
        return {"reply": f"I couldn't find any tickets for {email}."}
    if len(found) == 1:
        return {"reply": f"Here is your ticket for {email}:\n\n{_ticket_details(found[0])}"}

    listing = "\n".join(f"{i}. {t['id']}: {t['summary']} (Status: {t['status']})" for i, t in enumerate(found, 1))
    ans = _ask(f"I found {len(found)} tickets for {email}:\n{listing}\n\nWhich one would you like details on? Reply with the number or ticket ID.")
    chosen = _pick_ticket(found, ans)
    if not chosen:
        return {"reply": "I couldn't match that to one of your tickets. Ask me again and pick a number from the list."}
    return {"reply": _ticket_details(chosen)}


def smalltalk(state: AgentState):
    m = _norm(state["message"])
    if "thank" in m:
        return {"reply": "You're welcome! Let me know if there's anything else."}
    if "bye" in m:
        return {"reply": "Goodbye! Have a great day."}
    return {"reply": "Hello! I'm SmartDesk AI. I can answer IT and HR questions, raise a support ticket, or check the status of a ticket."}


def out_of_scope(state: AgentState):
    return {"reply": "I can only help with IT and HR questions, support tickets and ticket status at NovaTech. Is there anything like that I can help with?"}


# ---------------------------------------------------------------- routing
INTENT_TO_NODE = {
    "kb_query": "rag_answer",
    "create_ticket": "draft_ticket",
    "ticket_status": "ensure_email",
    "manage_email": "manage_email",
    "smalltalk": "smalltalk",
    "out_of_scope": "out_of_scope",
}


def route_intent(state: AgentState) -> str:
    return INTENT_TO_NODE[state["intent"]]


def after_rag(state: AgentState) -> str:
    return "offer_ticket" if state.get("escalate") else END


def after_offer(state: AgentState) -> str:
    if state.get("reroute"):
        return "classify"
    return "draft_ticket" if state.get("wants_ticket") else END


def after_draft(state: AgentState) -> str:
    return END if state.get("cancelled") else "ensure_email"


def after_email(state: AgentState) -> str:
    if state.get("cancelled"):
        return END
    return "ticket_status" if state.get("flow") == "status" else "confirm_ticket"


def after_confirm(state: AgentState) -> str:
    return "submit_ticket" if state.get("confirmed") else END


def build_graph():
    g = StateGraph(AgentState)
    for name, fn in [
        ("classify", classify), ("rag_answer", rag_answer), ("offer_ticket", offer_ticket),
        ("draft_ticket", draft_ticket), ("ensure_email", ensure_email), ("confirm_ticket", confirm_ticket),
        ("submit_ticket", submit_ticket), ("ticket_status", ticket_status),
        ("smalltalk", smalltalk), ("out_of_scope", out_of_scope), ("manage_email", manage_email),
    ]:
        g.add_node(name, fn)

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_intent, list(INTENT_TO_NODE.values()))
    g.add_conditional_edges("rag_answer", after_rag, ["offer_ticket", END])
    g.add_conditional_edges("offer_ticket", after_offer, ["classify", "draft_ticket", END])
    g.add_conditional_edges("draft_ticket", after_draft, ["ensure_email", END])
    g.add_conditional_edges("ensure_email", after_email, ["ticket_status", "confirm_ticket", END])
    g.add_conditional_edges("confirm_ticket", after_confirm, ["submit_ticket", END])
    for terminal in ("submit_ticket", "ticket_status", "smalltalk", "out_of_scope", "manage_email"):
        g.add_edge(terminal, END)

    return g.compile(checkpointer=MemorySaver())


graph = build_graph()

# ---------------------------------------------------------------- public API
_epoch: dict[str, int] = {}  # bump to start a fresh thread when the user clicks "New chat"


def _config(session_id: str) -> dict:
    return {"configurable": {"thread_id": f"{session_id}:{_epoch.get(session_id, 0)}"}}


def reset(session_id: str) -> None:
    _epoch[session_id] = _epoch.get(session_id, 0) + 1


def chat(session_id: str, message: str, user_id: str | None = None) -> dict:
    """Send one user message. Returns {"reply", "intent", "awaiting_input", "cached"}.

    user_id is the anonymous browser id. If we already know this user's email from an earlier
    session, it is loaded into the graph state, so the agent never has to ask for it again."""
    config = _config(session_id)
    config["run_name"] = "smartdesk_chat"  # label shown in LangSmith traces
    paused = bool(graph.get_state(config).next)  # True when the last turn ended on an interrupt
    if paused:
        graph.invoke(Command(resume=message), config)
    else:
        payload = {"message": message, "user_id": user_id}
        saved_email = profiles.get_email(user_id)
        if saved_email:
            payload["email"] = saved_email
        graph.invoke(payload, config)

    snap = graph.get_state(config)
    if snap.next:  # paused again: the interrupt payload is the question for the user
        prompts = [i.value for task in snap.tasks for i in task.interrupts]
        return {"reply": str(prompts[0]), "intent": snap.values.get("intent"), "awaiting_input": True, "cached": False}
    return {"reply": snap.values.get("reply") or "Sorry, I didn't get that.",
            "intent": snap.values.get("intent"), "awaiting_input": False,
            "cached": bool(snap.values.get("cached"))}


if __name__ == "__main__":
    if "--diagram" in sys.argv:
        print(graph.get_graph().draw_mermaid())
        sys.exit()
    sid = uuid.uuid4().hex
    print("SmartDesk AI (type 'exit' to quit). Demo tickets exist for demo@novatech.example\n")
    while True:
        text = input("You: ").strip()
        if text.lower() in ("exit", "quit"):
            break
        if text:
            print("Bot:", chat(sid, text)["reply"], "\n")
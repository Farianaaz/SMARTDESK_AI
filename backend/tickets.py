"""Ticket service.

USE_MOCK_JIRA=true  -> in-memory mock (good for UI testing)
USE_MOCK_JIRA=false -> real Jira, through the Atlassian MCP server (see jira_mcp.py)

Real-Jira design notes:
- Jira cannot be queried "by reporter email" for people without Jira accounts, so the employee's
  email is written into the ticket description AND saved in a local index (email -> ticket keys).
- Status lookups read the index, plus a JQL text search as a fallback (finds tickets that were
  created by hand in Jira, as long as the description contains the email), then fetch live status.
"""
import json
import os
import re
import threading
from itertools import count
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

USE_MOCK = os.getenv("USE_MOCK_JIRA", "true").lower() == "true"
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "SD")
ISSUE_TYPE = os.getenv("JIRA_ISSUE_TYPE", "Task")
JIRA_URL = os.getenv("JIRA_URL", "https://novatech.example").rstrip("/")
SET_PRIORITY = os.getenv("JIRA_SET_PRIORITY", "true").lower() == "true"
PRIORITY_MAP = {"Low": "Low", "Medium": "Medium", "High": "High", "Critical": "Highest"}
KEY_RE = re.compile(rf"\b{re.escape(PROJECT_KEY)}-\d+\b")
EMAIL_SAFE_RE = re.compile(r"^[\w.+-]+@[\w-]+\.[\w.-]+$")  # also keeps quotes out of the JQL string

INDEX_PATH = Path(__file__).parent / "ticket_index.json"
_index_lock = threading.Lock()


# ------------------------------------------------------------------ local email -> keys index
def _load_index() -> dict:
    try:
        return json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return {}


def _add_to_index(email: str, key: str) -> None:
    with _index_lock:
        idx = _load_index()
        idx.setdefault(email.lower(), [])
        if key not in idx[email.lower()]:
            idx[email.lower()].append(key)
        INDEX_PATH.write_text(json.dumps(idx, indent=2), encoding="utf-8")


# ------------------------------------------------------------------ real Jira via MCP
def _name(value):
    if isinstance(value, dict):
        return value.get("name") or value.get("value")
    return str(value) if value else None


def _create_jira(email, summary, description, category, priority) -> dict:
    import jira_mcp

    body = f"{description}\n\nReporter email: {email}\nCategory: {category}\nRaised via: SmartDesk AI"

    def attempt(with_priority: bool) -> str:
        fields = {"labels": ["smartdesk", category.lower()]}
        if with_priority:
            fields["priority"] = {"name": PRIORITY_MAP.get(priority, "Medium")}
        return jira_mcp.call_tool("jira_create_issue", {
            "project_key": PROJECT_KEY,
            "summary": summary,
            "issue_type": ISSUE_TYPE,
            "description": body,
            "additional_fields": json.dumps(fields),
        })

    try:
        text = attempt(SET_PRIORITY)
    except RuntimeError as e:
        # Some Jira projects do not allow setting priority on create: retry without it
        if SET_PRIORITY and "priority" in str(e).lower():
            text = attempt(False)
        else:
            raise
    match = KEY_RE.search(text)
    if not match:
        raise RuntimeError(f"Could not read the ticket key from the Jira response: {text[:200]}")
    key = match.group(0)
    _add_to_index(email, key)
    return {"id": key, "url": f"{JIRA_URL}/browse/{key}", "email": email.lower(), "summary": summary,
            "category": category, "priority": priority, "status": "Open", "comments": []}


def _get_issue(key: str) -> dict:
    import jira_mcp

    text = jira_mcp.call_tool("jira_get_issue", {"issue_key": key, "comment_limit": 3})
    try:
        data = json.loads(text)
    except ValueError:
        data = {}
    f = data.get("fields", data)  # tolerate flat or nested response shapes
    raw_comments = data.get("comments") or (f.get("comment") or {}).get("comments") or []
    comments = [c["body"] for c in raw_comments if isinstance(c, dict) and isinstance(c.get("body"), str)]
    return {
        "id": key,
        "url": f"{JIRA_URL}/browse/{key}",
        "summary": f.get("summary") or key,
        "status": _name(f.get("status")) or "Unknown",
        "priority": _name(f.get("priority")) or "Unspecified",
        "comments": comments,
    }


def _get_tickets_jira(email: str) -> list[dict]:
    import jira_mcp

    keys = list(reversed(_load_index().get(email.lower(), [])))  # newest first
    if EMAIL_SAFE_RE.match(email):
        jql = 'project = {} AND text ~ "\\"{}\\"" ORDER BY created DESC'.format(PROJECT_KEY, email)
        try:
            found = jira_mcp.call_tool("jira_search", {"jql": jql, "limit": 10, "fields": "summary"})
            keys += [k for k in KEY_RE.findall(found) if k not in keys]
        except RuntimeError:
            pass  # the local index still works if the search fails
    results = []
    for key in keys[:10]:
        try:
            results.append(_get_issue(key))
        except RuntimeError:
            continue  # deleted or inaccessible issue
    return results


# ------------------------------------------------------------------ mock
_counter = count(1001)
_store: list[dict] = []


def _create_mock(email, summary, description, category, priority) -> dict:
    ticket_id = f"{PROJECT_KEY}-{next(_counter)}"
    ticket = {"id": ticket_id, "url": f"{JIRA_URL}/browse/{ticket_id}", "email": email.lower(),
              "summary": summary, "description": description, "category": category,
              "priority": priority, "status": "Open", "comments": []}
    _store.append(ticket)
    return ticket


def _seed_demo_data() -> None:
    """Two pre-made tickets so you can test the multiple-ticket flow with demo@novatech.example."""
    for summary, status, comment in [
        ("VPN drops every 10 minutes", "In Progress", "IT team: investigating the VPN gateway logs."),
        ("Request access to the finance SharePoint site", "Resolved", "IT team: access granted by the site owner."),
    ]:
        t = _create_mock("demo@novatech.example", summary, summary, "IT", "Medium")
        t["status"] = status
        t["comments"].append(comment)


if USE_MOCK:
    _seed_demo_data()


# ------------------------------------------------------------------ public API (used by agent.py)
def create_ticket(email: str, summary: str, description: str, category: str, priority: str = "Medium") -> dict:
    if USE_MOCK:
        return _create_mock(email, summary, description, category, priority)
    return _create_jira(email, summary, description, category, priority)


def get_tickets(email: str) -> list[dict]:
    if USE_MOCK:
        return [t for t in _store if t["email"] == email.lower()]
    return _get_tickets_jira(email)
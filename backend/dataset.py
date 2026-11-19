"""Evaluation datasets. Each example is {"inputs": {...}, "outputs": {...}} (the LangSmith format).

Three suites:
  answerable   - questions the KB CAN answer; `expected_keywords` are facts the answer must contain
  unanswerable - questions the KB must NOT answer; the agent should escalate (offer a ticket)
  routing      - messages with the expected intent

Questions are paraphrases of the KB wording on purpose, so retrieval is really tested.
If you change the KB, update the keywords. Bump DATASET_VERSION when you change examples, because
LangSmith datasets are only created once per name.
"""
DATASET_VERSION = "v1"

ANSWERABLE = [
    ("I forgot my password, how do I get a new one?", ["password portal"]),
    ("How do I connect to the VPN?", ["software center"]),
    ("How many sick leave days do I get per year?", ["8 days"]),
    ("How many casual leave days do I get in a year?", ["12 days"]),
    ("How many earned leave days are there annually?", ["15 days"]),
    ("How long is paternity leave?", ["2 weeks"]),
    ("What is the duration of maternity leave?", ["26 weeks"]),
    ("Am I allowed to work from home?", ["2 days"]),
    ("What are the core working hours?", ["11 AM", "4 PM"]),
    ("When do I get paid each month?", ["last working day"]),
    ("How do I get my expenses reimbursed?", ["30 days"]),
    ("How does stock option vesting work?", ["cliff"]),
    ("How long is the probation period?", ["6 months"]),
    ("How long is the notice period after confirmation?", ["60 days"]),
    ("When will I receive my laptop?", ["first day"]),
    ("What time should I arrive on day one?", ["9:30"]),
    ("Who is covered by the company health insurance?", ["spouse"]),
    ("How long do I have to complete insurance enrolment?", ["30 days"]),
    ("What is the IT helpdesk extension?", ["2020"]),
    ("How can I contact the HR helpdesk by email?", ["hr@novatech.example"]),
    ("How do I get on the guest Wi-Fi?", ["NovaTech-Guest"]),
    ("How do I report a phishing email?", ["Report Phishing"]),
    ("My account got locked, what should I do?", ["15 minutes"]),
    ("What are the password requirements?", ["12 characters"]),
    ("How do I set up multi-factor authentication?", ["Authenticator"]),
    ("Can I plug a USB stick into my laptop?", ["blocked"]),
    ("How long do deleted files stay in the recycle bin?", ["30 days"]),
    ("How do I ask for a software licence?", ["Software Request"]),
    ("How many earned leave days can I carry forward?", ["30 days"]),
    ("What should I do if my laptop is lost or stolen?", ["2020"]),
]

# Topics deliberately NOT in the KB: the agent must escalate instead of answering
UNANSWERABLE = [
    "My monitor is flickering",
    "Where can I park my car at the office?",
    "My keyboard has stopped working",
    "The projector in the meeting room is not working",
    "I would like an ergonomic chair for my desk",
    "What is on the cafeteria menu today?",
    "How do I book a meeting room?",
    "Can I bring my dog to the office?",
    "How do I get a business card printed?",
]

ROUTING = [
    ("How do I reset my password?", "kb_query"),
    ("My laptop screen is cracked", "kb_query"),
    ("How many sick leaves do I get?", "kb_query"),
    ("How do I set up my work email on my phone?", "kb_query"),
    ("Please create a ticket for my broken chair", "create_ticket"),
    ("Raise a ticket for my VPN issue", "create_ticket"),
    ("I want to log a support ticket", "create_ticket"),
    ("Open a ticket for the broken projector", "create_ticket"),
    ("What is the status of my ticket?", "ticket_status"),
    ("Any update on my VPN ticket?", "ticket_status"),
    ("Did anyone look at my issue yet?", "ticket_status"),
    ("Hello", "smalltalk"),
    ("Thanks, that helped!", "smalltalk"),
    ("Good morning", "smalltalk"),
    ("What's the weather like today?", "out_of_scope"),
    ("Who won the cricket match yesterday?", "out_of_scope"),
    ("Write me a Python function to sort a list", "out_of_scope"),
    ("Forget my email address", "manage_email"),
    ("I want to change my saved email", "manage_email"),
    ("Please use priya@company.com as my email from now on", "manage_email"),
]


# Retrieval comparison (python -m evals.run_eval --compare-retrievers): where does each method win?
# keyword queries contain exact terms (codes, names, numbers): BM25 should help
KEYWORD_QUERIES = [
    ("error 809", ["809"]),
    ("5 failed sign-in attempts", ["5 failed"]),
    ("10 GB disk space", ["10 GB"]),
    ("security@novatech.example", ["security@novatech.example"]),
    ("passwords.novatech.example", ["passwords.novatech.example"]),
    ("NovaTech-Guest", ["NovaTech-Guest"]),
    ("Report Phishing button", ["Report Phishing"]),
    ("90 days backup restore", ["90 days"]),
    ("NovaTech Authenticator QR code", ["QR code"]),
    ("Software Center VPN client", ["software center"]),
]
# semantic queries are paraphrases with almost no shared words: dense search should help
SEMANTIC_QUERIES = [
    ("I can't remember my login credentials", ["password portal"]),
    ("my phone with the authenticator got stolen", ["reset your mfa"]),
    ("how many days off can I take when ill", ["8 days"]),
    ("how do I get money back for work purchases", ["expense portal"]),
    ("can I work remotely a few days", ["2 days"]),
    ("when will salary hit my bank account", ["last working day"]),
    ("what happens to my share options over time", ["cliff"]),
    ("are my dependents covered by medical insurance", ["spouse"]),
    ("what do I do on my first morning", ["9:30"]),
    ("how to add work mail to my mobile", ["outlook app"]),
]


def retrieval_groups():
    return {"standard": ANSWERABLE, "keyword": KEYWORD_QUERIES, "semantic": SEMANTIC_QUERIES}


def answerable_examples():
    return [{"inputs": {"question": q}, "outputs": {"expected_keywords": kw}} for q, kw in ANSWERABLE]


def unanswerable_examples():
    return [{"inputs": {"question": q}, "outputs": {"expected_escalate": True}} for q in UNANSWERABLE]


def routing_examples():
    return [{"inputs": {"message": m}, "outputs": {"intent": i}} for m, i in ROUTING]
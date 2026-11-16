"""LangSmith tracing helpers. Everything degrades to a no-op if `langsmith` is not installed
or tracing is switched off, so the app runs the same either way.

Turn tracing on with these variables in .env:
    LANGSMITH_TRACING=true
    LANGSMITH_API_KEY=...
    LANGSMITH_PROJECT=smartdesk-ai
"""
import os
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()  # must run before langsmith reads its environment variables

try:
    from langsmith import traceable  # decorator: records a function call as a trace
except ImportError:  # langsmith not installed: make it a pass-through decorator
    def traceable(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda fn: fn


def tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true" and bool(os.getenv("LANGSMITH_API_KEY"))


@lru_cache(maxsize=1)
def openai_client():
    """One shared OpenAI client. Wrapped so every LLM call (tokens, latency, cost) shows up in LangSmith."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        from langsmith.wrappers import wrap_openai

        return wrap_openai(client)
    except ImportError:
        return client
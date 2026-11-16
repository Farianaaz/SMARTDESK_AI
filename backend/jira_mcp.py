"""Synchronous wrapper around an MCP server (sooperset/mcp-atlassian) that exposes Jira tools.

The agent code is synchronous, but the MCP SDK is async. This module runs ONE long-lived worker
task on a background event loop. The worker starts the MCP server as a subprocess (stdio),
keeps the session open, and serves tool calls from a queue, so the server is started only once.

Usage:   text = call_tool("jira_search", {"jql": "project = SD"})
Debug:   python jira_mcp.py --tools      # list the tools the server exposes
         python jira_mcp.py --test       # run a JQL search against your project
"""
import asyncio
import concurrent.futures
import logging
import os
import sys
import threading
import time

from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

COMMAND = os.getenv("MCP_JIRA_COMMAND", "uvx")
ARGS = os.getenv("MCP_JIRA_ARGS", "mcp-atlassian").split()
STARTUP_TIMEOUT = 120  # first start can be slow while uvx downloads the server
CALL_TIMEOUT = 60
_LIST_TOOLS = "__list_tools__"


def _server_params():
    from mcp import StdioServerParameters

    # Secrets come from the environment (.env), never from code.
    env = {k: os.environ[k] for k in ("JIRA_URL", "JIRA_USERNAME", "JIRA_API_TOKEN") if k in os.environ}
    return StdioServerParameters(command=COMMAND, args=ARGS, env=env)


class _MCPBridge:
    def __init__(self):
        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True, name="mcp-loop").start()
        self._queue = asyncio.Queue()
        self._ready = threading.Event()
        self._worker = None
        self._start_lock = threading.Lock()

    async def _run_worker(self):
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        try:
            async with stdio_client(_server_params()) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._ready.set()
                    log.info("MCP Jira server connected")
                    while True:
                        tool, args, fut = await self._queue.get()
                        if fut.cancelled():
                            continue
                        try:
                            if tool == _LIST_TOOLS:
                                fut.set_result(await session.list_tools())
                            else:
                                fut.set_result(await session.call_tool(tool, args))
                        except Exception as e:
                            fut.set_exception(e)
                            raise  # connection is probably broken: end the worker so the next call restarts it
        finally:
            self._ready.clear()
            while not self._queue.empty():
                _, _, fut = self._queue.get_nowait()
                if not fut.done():
                    fut.set_exception(RuntimeError("MCP server connection lost"))

    def _ensure_worker(self):
        with self._start_lock:
            if self._worker is None or self._worker.done():
                self._ready.clear()
                self._worker = asyncio.run_coroutine_threadsafe(self._run_worker(), self._loop)
            deadline = time.time() + STARTUP_TIMEOUT
            while not self._ready.wait(0.2):
                if self._worker.done():
                    raise RuntimeError(f"MCP server failed to start: {self._worker.exception()}")
                if time.time() > deadline:
                    raise TimeoutError("MCP server start timed out")

    def request(self, tool: str, arguments: dict):
        self._ensure_worker()
        fut = concurrent.futures.Future()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, (tool, arguments, fut))
        return fut.result(timeout=CALL_TIMEOUT)


_bridge = None
_bridge_lock = threading.Lock()


def _get_bridge() -> _MCPBridge:
    global _bridge
    with _bridge_lock:
        if _bridge is None:
            _bridge = _MCPBridge()
        return _bridge


def _field(obj, camel: str, snake: str, default=None):
    """Newer MCP SDK versions renamed camelCase attributes to snake_case (isError -> is_error,
    inputSchema -> input_schema). Accept both so the code works with either version."""
    for name in (camel, snake):
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def call_tool(name: str, arguments: dict) -> str:
    """Call an MCP tool and return its text output. Raises RuntimeError if the tool reports an error."""
    result = _get_bridge().request(name, arguments)
    text = "\n".join(c.text for c in result.content if getattr(c, "text", None))
    if _field(result, "isError", "is_error", False):
        raise RuntimeError(text or f"MCP tool {name} failed")
    return text


def list_tools():
    return _get_bridge().request(_LIST_TOOLS, {}).tools


def warm_up() -> None:
    """Start the server in the background at app startup so the first user request is not slow."""
    try:
        _get_bridge()._ensure_worker()
    except Exception as e:
        log.warning("MCP warm-up failed: %s", e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if "--tools" in sys.argv:
        for t in list_tools():
            if t.name.startswith("jira_") and t.name in (
                "jira_create_issue", "jira_get_issue", "jira_search", "jira_get_transitions"
            ):
                schema = _field(t, "inputSchema", "input_schema", {}) or {}
                props = schema.get("properties", {})
                print(f"\n{t.name}  required={schema.get('required', [])}")
                for p, spec in props.items():
                    print(f"   - {p}: {spec.get('type', spec.get('anyOf', ''))}")
        print("\nAll tool names:", ", ".join(t.name for t in list_tools()))
    elif "--test" in sys.argv:
        key = os.getenv("JIRA_PROJECT_KEY", "SD")
        print(call_tool("jira_search", {"jql": f"project = {key} ORDER BY created DESC", "limit": 3}))
    else:
        print(__doc__)
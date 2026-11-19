"""Persistent employee profile, so the email is asked ONCE, not in every session.

The browser keeps a random `user_id` in localStorage and sends it with every message.
This module maps user_id -> email in a small SQLite file (profiles.db).
No passwords or other personal data are stored. The user can say "forget my email" at any time.
"""
import os
import re
import sqlite3
import threading
import time
from contextlib import closing
from pathlib import Path

DB_PATH = Path(os.getenv("PROFILE_DB", Path(__file__).parent / "profiles.db"))
_lock = threading.Lock()
_USER_ID_RE = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS profiles ("
        "user_id TEXT PRIMARY KEY, email TEXT NOT NULL, updated_at REAL NOT NULL)"
    )
    return conn


def _valid(user_id) -> bool:
    return bool(user_id) and bool(_USER_ID_RE.match(str(user_id)))


def get_email(user_id) -> str | None:
    if not _valid(user_id):
        return None
    with _lock, closing(_connect()) as conn:
        row = conn.execute("SELECT email FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
    return row[0] if row else None


def set_email(user_id, email: str) -> None:
    if not _valid(user_id) or not email:
        return
    with _lock, closing(_connect()) as conn, conn:
        conn.execute(
            "INSERT INTO profiles (user_id, email, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(user_id) DO UPDATE SET email = excluded.email, updated_at = excluded.updated_at",
            (user_id, email.lower(), time.time()),
        )


def forget(user_id) -> None:
    if not _valid(user_id):
        return
    with _lock, closing(_connect()) as conn, conn:
        conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
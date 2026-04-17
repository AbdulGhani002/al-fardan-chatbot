"""SQLite helpers for chat history + unanswered queries.

SQLite keeps the VPS footprint tiny and avoids needing a separate DB
process. When volume outgrows SQLite we migrate to the same MongoDB
the CRM already uses (mongodb-alfardan on :27018).
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_messages (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  session_token   TEXT    NOT NULL,
  role            TEXT    NOT NULL,   -- user | bot | system
  text            TEXT    NOT NULL,
  match_type      TEXT,               -- kb_hit, low_confidence, etc.
  matched_entry   TEXT,
  match_score     REAL,
  user_id         TEXT,               -- CRM user _id if known
  user_email      TEXT,
  created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_messages_session ON chat_messages(session_token, id);

CREATE TABLE IF NOT EXISTS unanswered_queries (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  session_token   TEXT    NOT NULL,
  user_text       TEXT    NOT NULL,
  best_score      REAL    NOT NULL DEFAULT 0,
  nearest_entry   TEXT,
  user_id         TEXT,
  user_email      TEXT,
  status          TEXT    NOT NULL DEFAULT 'new',  -- new, promoted, resolved, dismissed
  review_notes    TEXT,
  created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_unanswered_status ON unanswered_queries(status, id DESC);
"""


def ensure_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    ensure_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ─── Chat history ──────────────────────────────────────────────────────

def record_message(
    db_path: Path,
    session_token: str,
    role: str,
    text: str,
    match_type: Optional[str] = None,
    matched_entry: Optional[str] = None,
    match_score: Optional[float] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO chat_messages
              (session_token, role, text, match_type, matched_entry,
               match_score, user_id, user_email, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_token,
                role,
                text,
                match_type,
                matched_entry,
                match_score,
                user_id,
                user_email,
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
        return int(cur.lastrowid or 0)


def get_history(
    db_path: Path, session_token: str, limit: int = 50
) -> list[dict]:
    with connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT role, text, created_at, match_type, match_score
            FROM chat_messages
            WHERE session_token = ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (session_token, limit),
        ).fetchall()
        return [dict(r) for r in rows]


# ─── Unanswered queries ────────────────────────────────────────────────

def capture_unanswered(
    db_path: Path,
    session_token: str,
    user_text: str,
    best_score: float,
    nearest_entry: Optional[str] = None,
    user_id: Optional[str] = None,
    user_email: Optional[str] = None,
) -> int:
    with connect(db_path) as conn:
        cur = conn.execute(
            """
            INSERT INTO unanswered_queries
              (session_token, user_text, best_score, nearest_entry,
               user_id, user_email, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, 'new', ?)
            """,
            (
                session_token,
                user_text,
                best_score,
                nearest_entry,
                user_id,
                user_email,
                datetime.utcnow().isoformat(timespec="seconds"),
            ),
        )
        return int(cur.lastrowid or 0)


def list_unanswered(
    db_path: Path, status: str = "new", limit: int = 100
) -> list[dict]:
    with connect(db_path) as conn:
        if status == "all":
            rows = conn.execute(
                """SELECT * FROM unanswered_queries
                   ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM unanswered_queries
                   WHERE status = ?
                   ORDER BY id DESC LIMIT ?""",
                (status, limit),
            ).fetchall()
        return [dict(r) for r in rows]


def update_query_status(
    db_path: Path, qid: int, status: str, review_notes: str = ""
) -> bool:
    with connect(db_path) as conn:
        cur = conn.execute(
            """UPDATE unanswered_queries
               SET status = ?, review_notes = ?
               WHERE id = ?""",
            (status, review_notes, qid),
        )
        return cur.rowcount > 0


# ─── Training data export ─────────────────────────────────────────────

def list_all_messages(
    db_path: Path,
    since_iso: Optional[str] = None,
    limit: int = 10_000,
) -> list[dict]:
    """Pull every chat message (paired user + bot turns) for training.

    Ordered by session_token then id so each session's conversation
    stays together in chronological order — ready to be converted into
    fine-tuning pairs or used to identify KB gaps.
    """
    with connect(db_path) as conn:
        if since_iso:
            rows = conn.execute(
                """SELECT id, session_token, role, text, match_type,
                          matched_entry, match_score, user_id, user_email,
                          created_at
                   FROM chat_messages
                   WHERE created_at >= ?
                   ORDER BY session_token, id
                   LIMIT ?""",
                (since_iso, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT id, session_token, role, text, match_type,
                          matched_entry, match_score, user_id, user_email,
                          created_at
                   FROM chat_messages
                   ORDER BY session_token, id
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def training_turn_pairs(
    db_path: Path, since_iso: Optional[str] = None, limit: int = 5000
) -> list[dict]:
    """Walk the message log and pair each 'user' turn with the
    immediately-following 'bot' turn. Drops dangling turns.

    Emits records shaped:
        {
          "user": "...",
          "bot": "...",
          "match_type": "kb_hit" | "low_confidence" | ...,
          "matched_entry": "crm-001" | null,
          "score": 0.84 | null,
          "session_token": "...",
          "created_at": "2026-04-17T..."
        }

    This is the format you'd hand to a curation workflow: look at
    low_confidence turns, decide whether to add them to the KB, or
    tune aliases on the existing `matched_entry`.
    """
    msgs = list_all_messages(db_path, since_iso=since_iso, limit=limit * 3)
    pairs: list[dict] = []
    # Iterate linearly; emit a pair whenever a user turn is followed
    # directly by a bot turn within the same session.
    for i in range(len(msgs) - 1):
        a = msgs[i]
        b = msgs[i + 1]
        if (
            a["role"] == "user"
            and b["role"] == "bot"
            and a["session_token"] == b["session_token"]
        ):
            pairs.append(
                {
                    "user": a["text"],
                    "bot": b["text"],
                    "match_type": b.get("match_type"),
                    "matched_entry": b.get("matched_entry"),
                    "score": b.get("match_score"),
                    "session_token": a["session_token"],
                    "created_at": a["created_at"],
                }
            )
            if len(pairs) >= limit:
                break
    return pairs


def message_stats(db_path: Path) -> dict:
    """Quick summary counters — rendered in /admin/conversations/stats."""
    with connect(db_path) as conn:
        total = conn.execute("SELECT COUNT(*) c FROM chat_messages").fetchone()["c"]
        user = conn.execute(
            "SELECT COUNT(*) c FROM chat_messages WHERE role = 'user'"
        ).fetchone()["c"]
        bot = conn.execute(
            "SELECT COUNT(*) c FROM chat_messages WHERE role = 'bot'"
        ).fetchone()["c"]
        sessions = conn.execute(
            "SELECT COUNT(DISTINCT session_token) c FROM chat_messages"
        ).fetchone()["c"]
        unanswered = conn.execute(
            "SELECT COUNT(*) c FROM unanswered_queries WHERE status = 'new'"
        ).fetchone()["c"]
        low_conf = conn.execute(
            """SELECT COUNT(*) c FROM chat_messages
               WHERE role = 'bot' AND match_type = 'low_confidence'"""
        ).fetchone()["c"]
    return {
        "total_messages": total,
        "user_messages": user,
        "bot_messages": bot,
        "unique_sessions": sessions,
        "unanswered_queue_new": unanswered,
        "low_confidence_bot_turns": low_conf,
    }

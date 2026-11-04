"""Local SQLite persistence for named conversations."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone

from .config import SQLITE_PATH


def connect():
    """Open the history database and ensure its tables exist."""
    database = sqlite3.connect(SQLITE_PATH)
    database.execute(
        "CREATE TABLE IF NOT EXISTS conversations "
        "(id TEXT PRIMARY KEY, name TEXT NOT NULL, updated TEXT NOT NULL)"
    )
    database.execute(
        "CREATE TABLE IF NOT EXISTS messages "
        "(id INTEGER PRIMARY KEY, conversation_id TEXT NOT NULL, "
        "role TEXT NOT NULL, content TEXT NOT NULL, sources TEXT NOT NULL "
        "DEFAULT '[]', provider TEXT, model TEXT, latency REAL, "
        "FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE)"
    )
    database.commit()
    return database


def list_conversations():
    with connect() as database:
        return database.execute(
            "SELECT id, name, updated FROM conversations ORDER BY updated DESC"
        ).fetchall()


def create_conversation(name="New conversation"):
    conversation_id = uuid.uuid4().hex
    updated = datetime.now(timezone.utc).isoformat()
    with connect() as database:
        database.execute(
            "INSERT INTO conversations VALUES (?, ?, ?)",
            (conversation_id, name, updated),
        )
    return conversation_id


def rename_conversation(conversation_id, name):
    updated = datetime.now(timezone.utc).isoformat()
    with connect() as database:
        database.execute(
            "UPDATE conversations SET name = ?, updated = ? WHERE id = ?",
            (name[:80], updated, conversation_id),
        )


def delete_conversation(conversation_id):
    with connect() as database:
        database.execute("PRAGMA foreign_keys = ON")
        database.execute(
            "DELETE FROM messages WHERE conversation_id = ?", (conversation_id,)
        )
        database.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


def get_messages(conversation_id):
    with connect() as database:
        rows = database.execute(
            "SELECT role, content, sources, provider, model, latency "
            "FROM messages WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        ).fetchall()
    return [
        {
            "role": row[0],
            "content": row[1],
            "sources": json.loads(row[2]),
            "provider": row[3],
            "model": row[4],
            "latency": row[5],
        }
        for row in rows
    ]


def add_message(
    conversation_id,
    role,
    content,
    sources=None,
    provider=None,
    model=None,
    latency=None,
):
    updated = datetime.now(timezone.utc).isoformat()
    with connect() as database:
        database.execute(
            "INSERT INTO messages(conversation_id, role, content, sources, "
            "provider, model, latency) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                conversation_id,
                role,
                content,
                json.dumps(sources or []),
                provider,
                model,
                latency,
            ),
        )
        database.execute(
            "UPDATE conversations SET updated = ? WHERE id = ?",
            (updated, conversation_id),
        )
        conversation = database.execute(
            "SELECT name FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        if role == "user" and conversation and conversation[0] == "New conversation":
            database.execute(
                "UPDATE conversations SET name = ? WHERE id = ?",
                (content[:48], conversation_id),
            )

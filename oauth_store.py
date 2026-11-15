"""SQLite storage for the OAuth authorization server (same module as ga4-mcp-server).

ebay-platform keeps this state in Postgres. This server only ever stores OAuth
bookkeeping -- registered clients, short-lived auth codes, access/refresh
tokens -- so a single file on the Coolify volume is enough and saves us a
database dependency entirely.

No Klaviyo API keys are stored here; those live only in environment variables.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any, Optional

DATA_DIR = os.environ.get("DATA_DIR", "/data")
DB_PATH = os.environ.get("DB_PATH") or os.path.join(DATA_DIR, "seo.db")

_local = threading.local()


def _conn() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # WAL keeps reads from blocking the token endpoint during writes.
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=5000;")
        _local.conn = conn
    return conn


def init_schema() -> None:
    c = _conn()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS oauth_clients (
            client_id                  TEXT PRIMARY KEY,
            client_secret              TEXT,
            client_name                TEXT,
            redirect_uris              TEXT NOT NULL,
            grant_types                TEXT,
            response_types             TEXT,
            token_endpoint_auth_method TEXT,
            scope                      TEXT,
            created_at                 TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS oauth_auth_codes (
            code                  TEXT PRIMARY KEY,
            client_id             TEXT NOT NULL,
            redirect_uri          TEXT NOT NULL,
            scope                 TEXT,
            code_challenge        TEXT,
            code_challenge_method TEXT,
            expires_at            TEXT NOT NULL,
            used                  INTEGER NOT NULL DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS oauth_access_tokens (
            token                     TEXT PRIMARY KEY,
            client_id                 TEXT NOT NULL,
            scope                     TEXT,
            expires_at                TEXT NOT NULL,
            refresh_token             TEXT,
            refresh_token_expires_at  TEXT,
            created_at                TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_oauth_access_refresh
            ON oauth_access_tokens(refresh_token);
        """
    )
    c.commit()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# --------------------------------------------------------------------------
# clients
# --------------------------------------------------------------------------

def oauth_client_create(
    client_id: str,
    client_secret: str | None,
    client_name: str | None,
    redirect_uris: list[str],
    grant_types: list[str],
    response_types: list[str],
    token_endpoint_auth_method: str,
    scope: str,
) -> None:
    c = _conn()
    c.execute(
        """INSERT OR REPLACE INTO oauth_clients
           (client_id, client_secret, client_name, redirect_uris, grant_types,
            response_types, token_endpoint_auth_method, scope, created_at)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        (
            client_id,
            client_secret,
            client_name,
            json.dumps(redirect_uris),
            json.dumps(grant_types),
            json.dumps(response_types),
            token_endpoint_auth_method,
            scope,
            _iso(_now()),
        ),
    )
    c.commit()


def oauth_client_get(client_id: str) -> Optional[dict[str, Any]]:
    row = _conn().execute(
        "SELECT * FROM oauth_clients WHERE client_id = ?", (client_id,)
    ).fetchone()
    if not row:
        return None
    return {
        "client_id": row["client_id"],
        "client_secret": row["client_secret"],
        "client_name": row["client_name"],
        "redirect_uris": json.loads(row["redirect_uris"]),
        "grant_types": json.loads(row["grant_types"] or "[]"),
        "response_types": json.loads(row["response_types"] or "[]"),
        "token_endpoint_auth_method": row["token_endpoint_auth_method"] or "client_secret_post",
        "scope": row["scope"],
    }


# --------------------------------------------------------------------------
# authorization codes
# --------------------------------------------------------------------------

def oauth_auth_code_create(
    code: str,
    client_id: str,
    redirect_uri: str,
    scope: str,
    code_challenge: str | None,
    code_challenge_method: str | None,
    expires_at: datetime,
) -> None:
    c = _conn()
    c.execute(
        """INSERT INTO oauth_auth_codes
           (code, client_id, redirect_uri, scope, code_challenge, code_challenge_method, expires_at)
           VALUES (?,?,?,?,?,?,?)""",
        (code, client_id, redirect_uri, scope, code_challenge, code_challenge_method, _iso(expires_at)),
    )
    c.commit()


def oauth_auth_code_consume(code: str) -> Optional[dict[str, Any]]:
    """Atomically mark a code used and return it, or None if already spent."""
    c = _conn()
    cur = c.execute(
        "UPDATE oauth_auth_codes SET used = 1 WHERE code = ? AND used = 0 RETURNING *",
        (code,),
    )
    row = cur.fetchone()
    c.commit()
    if not row:
        return None
    expires = _parse(row["expires_at"])
    if expires and expires < _now():
        return None
    return {
        "code": row["code"],
        "client_id": row["client_id"],
        "redirect_uri": row["redirect_uri"],
        "scope": row["scope"],
        "code_challenge": row["code_challenge"],
        "code_challenge_method": row["code_challenge_method"],
    }


# --------------------------------------------------------------------------
# access + refresh tokens
# --------------------------------------------------------------------------

def oauth_access_token_create(
    token: str,
    client_id: str,
    scope: str,
    expires_at: datetime,
    refresh_token: str,
    refresh_expires_at: datetime,
) -> None:
    c = _conn()
    c.execute(
        """INSERT INTO oauth_access_tokens
           (token, client_id, scope, expires_at, refresh_token, refresh_token_expires_at, created_at)
           VALUES (?,?,?,?,?,?,?)""",
        (token, client_id, scope, _iso(expires_at), refresh_token, _iso(refresh_expires_at), _iso(_now())),
    )
    c.commit()


def oauth_access_token_validate(token: str) -> Optional[dict[str, Any]]:
    row = _conn().execute(
        "SELECT * FROM oauth_access_tokens WHERE token = ?", (token,)
    ).fetchone()
    if not row:
        return None
    expires = _parse(row["expires_at"])
    if expires and expires < _now():
        return None
    return {"client_id": row["client_id"], "scope": row["scope"]}


def oauth_refresh_token_lookup(refresh_token: str) -> Optional[dict[str, Any]]:
    row = _conn().execute(
        "SELECT * FROM oauth_access_tokens WHERE refresh_token = ?", (refresh_token,)
    ).fetchone()
    if not row:
        return None
    return {
        "client_id": row["client_id"],
        "scope": row["scope"],
        "refresh_expires_at": _parse(row["refresh_token_expires_at"]),
    }


def purge_expired() -> int:
    """Housekeeping: drop spent codes and dead tokens. Called at startup."""
    c = _conn()
    now = _iso(_now())
    removed = c.execute("DELETE FROM oauth_auth_codes WHERE expires_at < ? OR used = 1", (now,)).rowcount
    removed += c.execute(
        "DELETE FROM oauth_access_tokens WHERE refresh_token_expires_at IS NOT NULL "
        "AND refresh_token_expires_at < ?",
        (now,),
    ).rowcount
    c.commit()
    return max(removed, 0)

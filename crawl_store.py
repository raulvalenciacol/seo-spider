"""SQLite storage for crawls run through the MCP server.

Shares the DB file with the OAuth tables (oauth_store) so one Coolify volume
holds everything. Pages keep the full Screaming-Frog-style row as JSON plus a
few indexed columns the issue queries filter on.
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
        conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        _local.conn = conn
    return conn


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def init_schema() -> None:
    c = _conn()
    c.executescript(
        """
        CREATE TABLE IF NOT EXISTS crawls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            domain      TEXT NOT NULL,
            start_url   TEXT NOT NULL,
            status      TEXT NOT NULL,          -- running | done | failed | stopped
            settings    TEXT NOT NULL,          -- JSON
            progress    TEXT,                   -- last progress message
            crawled     INTEGER NOT NULL DEFAULT 0,
            pages       INTEGER NOT NULL DEFAULT 0,
            images      INTEGER NOT NULL DEFAULT 0,
            blocked     INTEGER NOT NULL DEFAULT 0,
            error       TEXT,
            started_at  TEXT NOT NULL,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_crawls_domain ON crawls(domain, started_at);

        CREATE TABLE IF NOT EXISTS crawl_pages (
            crawl_id      INTEGER NOT NULL REFERENCES crawls(id) ON DELETE CASCADE,
            url           TEXT NOT NULL,
            status_code   INTEGER,
            content_type  TEXT,
            indexability  TEXT,
            title         TEXT,
            meta_desc     TEXT,
            h1            TEXT,
            canonical     TEXT,
            redirect_url  TEXT,
            word_count    INTEGER,
            inlinks       INTEGER,
            crawl_depth   INTEGER,
            row           TEXT NOT NULL,        -- full JSON row
            PRIMARY KEY (crawl_id, url)
        );
        CREATE INDEX IF NOT EXISTS idx_pages_status ON crawl_pages(crawl_id, status_code);

        CREATE TABLE IF NOT EXISTS crawl_images (
            crawl_id   INTEGER NOT NULL REFERENCES crawls(id) ON DELETE CASCADE,
            page_url   TEXT NOT NULL,
            image_url  TEXT NOT NULL,
            alt_status TEXT NOT NULL,           -- Missing | Empty | Present
            alt_text   TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_images_crawl ON crawl_images(crawl_id, alt_status);

        CREATE TABLE IF NOT EXISTS site_headers (
            domain     TEXT PRIMARY KEY,
            headers    TEXT NOT NULL,           -- JSON {name: value}
            note       TEXT,
            expires_at TEXT,
            updated_at TEXT NOT NULL
        );
        """
    )
    c.commit()


# ---------- crawls ----------

def crawl_create(domain: str, start_url: str, settings: dict) -> int:
    c = _conn()
    cur = c.execute(
        "INSERT INTO crawls (domain, start_url, status, settings, started_at) VALUES (?, ?, 'running', ?, ?)",
        (domain, start_url, json.dumps(settings), _now()),
    )
    c.commit()
    return int(cur.lastrowid)


def crawl_progress(crawl_id: int, crawled: int, message: str) -> None:
    c = _conn()
    c.execute("UPDATE crawls SET crawled = ?, progress = ? WHERE id = ?", (crawled, message[:300], crawl_id))
    c.commit()


def crawl_finish(crawl_id: int, status: str, rows: list[dict], images: list[dict], blocked: int, error: str | None = None) -> None:
    c = _conn()
    with c:
        c.execute("DELETE FROM crawl_pages WHERE crawl_id = ?", (crawl_id,))
        c.execute("DELETE FROM crawl_images WHERE crawl_id = ?", (crawl_id,))
        c.executemany(
            """INSERT OR REPLACE INTO crawl_pages
               (crawl_id, url, status_code, content_type, indexability, title, meta_desc, h1, canonical,
                redirect_url, word_count, inlinks, crawl_depth, row)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    crawl_id, r.get("Address"), _int(r.get("Status Code")), r.get("Content Type"),
                    r.get("Indexability"), r.get("Title 1"), r.get("Meta Description 1"), r.get("H1-1"),
                    r.get("Canonical Link Element 1"), r.get("Redirect URL"), _int(r.get("Word Count")),
                    _int(r.get("Inlinks")), _int(r.get("Crawl Depth")), json.dumps(r, default=str),
                )
                for r in rows
            ],
        )
        c.executemany(
            "INSERT INTO crawl_images (crawl_id, page_url, image_url, alt_status, alt_text) VALUES (?, ?, ?, ?, ?)",
            [(crawl_id, i.get("Page URL"), i.get("Image URL"), i.get("Alt Text Status"), i.get("Alt Text")) for i in images],
        )
        c.execute(
            """UPDATE crawls SET status = ?, pages = ?, images = ?, blocked = ?, error = ?,
               crawled = ?, finished_at = ? WHERE id = ?""",
            (status, len(rows), len(images), blocked, error, len(rows), _now(), crawl_id),
        )


def crawl_fail(crawl_id: int, error: str) -> None:
    c = _conn()
    c.execute("UPDATE crawls SET status = 'failed', error = ?, finished_at = ? WHERE id = ?",
              (error[:1000], _now(), crawl_id))
    c.commit()


def crawl_get(crawl_id: int) -> Optional[dict[str, Any]]:
    row = _conn().execute("SELECT * FROM crawls WHERE id = ?", (crawl_id,)).fetchone()
    return _crawl_dict(row) if row else None


def crawl_list(domain: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
    if domain:
        rows = _conn().execute(
            "SELECT * FROM crawls WHERE domain = ? ORDER BY id DESC LIMIT ?", (domain, limit)).fetchall()
    else:
        rows = _conn().execute("SELECT * FROM crawls ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [_crawl_dict(r) for r in rows]


def crawl_running() -> Optional[dict[str, Any]]:
    row = _conn().execute("SELECT * FROM crawls WHERE status = 'running' ORDER BY id DESC LIMIT 1").fetchone()
    return _crawl_dict(row) if row else None


def crawl_delete(crawl_id: int) -> bool:
    c = _conn()
    with c:
        c.execute("DELETE FROM crawl_pages WHERE crawl_id = ?", (crawl_id,))
        c.execute("DELETE FROM crawl_images WHERE crawl_id = ?", (crawl_id,))
        cur = c.execute("DELETE FROM crawls WHERE id = ?", (crawl_id,))
    return cur.rowcount > 0


def _crawl_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["settings"] = json.loads(d.get("settings") or "{}")
    return d


# ---------- pages ----------

def pages(crawl_id: int) -> list[dict[str, Any]]:
    rows = _conn().execute("SELECT row FROM crawl_pages WHERE crawl_id = ? ORDER BY url", (crawl_id,)).fetchall()
    return [json.loads(r["row"]) for r in rows]


def page(crawl_id: int, url: str) -> Optional[dict[str, Any]]:
    row = _conn().execute("SELECT row FROM crawl_pages WHERE crawl_id = ? AND url = ?", (crawl_id, url)).fetchone()
    return json.loads(row["row"]) if row else None


def images(crawl_id: int, alt_status: str | None = None) -> list[dict[str, Any]]:
    if alt_status:
        rows = _conn().execute(
            "SELECT page_url, image_url, alt_status, alt_text FROM crawl_images WHERE crawl_id = ? AND alt_status = ?",
            (crawl_id, alt_status)).fetchall()
    else:
        rows = _conn().execute(
            "SELECT page_url, image_url, alt_status, alt_text FROM crawl_images WHERE crawl_id = ?", (crawl_id,)).fetchall()
    return [dict(r) for r in rows]


# ---------- per-site request headers (Shopify crawler signatures) ----------

def site_headers_set(domain: str, headers: dict, note: str | None, expires_at: str | None) -> None:
    c = _conn()
    c.execute(
        """INSERT INTO site_headers (domain, headers, note, expires_at, updated_at) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(domain) DO UPDATE SET headers = excluded.headers, note = excluded.note,
             expires_at = excluded.expires_at, updated_at = excluded.updated_at""",
        (domain, json.dumps(headers), note, expires_at, _now()),
    )
    c.commit()


def site_headers_get(domain: str) -> Optional[dict[str, Any]]:
    row = _conn().execute("SELECT * FROM site_headers WHERE domain = ?", (domain,)).fetchone()
    if not row:
        return None
    d = dict(row)
    d["headers"] = json.loads(d["headers"])
    return d


def site_headers_list() -> list[dict[str, Any]]:
    rows = _conn().execute("SELECT domain, note, expires_at, updated_at FROM site_headers ORDER BY domain").fetchall()
    return [dict(r) for r in rows]


def site_headers_delete(domain: str) -> bool:
    c = _conn()
    cur = c.execute("DELETE FROM site_headers WHERE domain = ?", (domain,))
    c.commit()
    return cur.rowcount > 0


def _int(v) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

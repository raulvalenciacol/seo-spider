"""SEO Spider MCP server: crawl a site from Claude / ChatGPT and query the results.

Crawls run in a background thread and are stored in SQLite (DATA_DIR/seo.db),
so a client starts a crawl, polls its status, then asks for issues, pages or a
comparison with an earlier crawl of the same domain. One crawl runs at a time
to keep memory bounded on the shared VPS.

Auth: static MCP_BEARER_TOKEN (Authorization: Bearer or X-API-Key) or OAuth
2.1 tokens issued by oauth_server (consent approved with STAFF_ACCESS_KEY).
"""
from __future__ import annotations

import logging
import os
import threading
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Mount, Route

import crawl_store as store
import oauth_server
import oauth_store
from spider_core import get_domain, parse_header_lines, run_spider

log = logging.getLogger("seo-mcp")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

MAX_PAGES_CAP = int(os.environ.get("CRAWL_MAX_PAGES_CAP", "2000"))
MAX_THREADS_CAP = int(os.environ.get("CRAWL_MAX_THREADS_CAP", "4"))

_crawl_lock = threading.Lock()
_active: dict[str, Any] = {"crawl_id": None, "stop": None}

mcp = FastMCP(
    "seo-spider",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
    stateless_http=True,
    json_response=True,
)


# ---------- helpers ----------

def _norm_start_url(url: str) -> str:
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url


def _domain_key(url_or_domain: str) -> str:
    d = url_or_domain.strip()
    if "://" in d:
        d = urlparse(d).netloc
    return d.lower().removeprefix("www.")


def _crawl_or_error(crawl_id: int) -> dict:
    c = store.crawl_get(crawl_id)
    if not c:
        raise ValueError(f"No crawl with id {crawl_id}. Use list_crawls to find one.")
    return c


def _html_pages(crawl_id: int) -> list[dict]:
    return [p for p in store.pages(crawl_id) if "text/html" in str(p.get("Content Type") or "")]


def _brief(p: dict, *extra: str) -> dict:
    out = {"url": p.get("Address"), "status": p.get("Status Code"), "title": p.get("Title 1")}
    for k in extra:
        out[k] = p.get(k)
    return out


def _run_crawl(crawl_id: int, start_url: str, settings: dict, stop: threading.Event) -> None:
    def progress(crawled: int, total: int, message: str) -> None:
        store.crawl_progress(crawl_id, crawled, message)

    try:
        rows, images, blocked = run_spider(
            start_url,
            settings["max_pages"],
            settings["threads"],
            settings["delay"],
            settings["scope"],
            proxy_url=None,
            use_spider_ua=settings["spider_user_agent"],
            extra_headers=settings.get("_headers") or None,
            progress=progress,
            should_stop=stop.is_set,
        )
        store.crawl_finish(crawl_id, "stopped" if stop.is_set() else "done", rows, images, blocked)
        log.info("crawl %s finished: %d pages, %d images, %d blocked", crawl_id, len(rows), len(images), blocked)
    except Exception as exc:  # noqa: BLE001
        log.exception("crawl %s failed", crawl_id)
        store.crawl_fail(crawl_id, str(exc))
    finally:
        with _crawl_lock:
            if _active["crawl_id"] == crawl_id:
                _active["crawl_id"] = None
                _active["stop"] = None


# ---------- tools: running crawls ----------

@mcp.tool()
def start_crawl(
    url: str,
    max_pages: int = 300,
    scope: str = "all",
    threads: int = 2,
    delay_seconds: float = 1.0,
    spider_user_agent: bool = False,
) -> dict:
    """Start crawling a website in the background and return a crawl_id.

    Poll crawl_status(crawl_id) until status is 'done', then use
    crawl_summary / crawl_issues / list_pages / compare_crawls.
    scope: 'all', 'products', 'collections' or 'products_and_collections'
    (the last three only keep Shopify /products/ and /collections/ URLs).
    Headers saved with set_site_headers for this domain (e.g. a Shopify
    crawler signature) are sent automatically. Only one crawl runs at a time.
    """
    start_url = _norm_start_url(url)
    if scope not in ("all", "products", "collections", "products_and_collections"):
        raise ValueError("scope must be all, products, collections or products_and_collections")
    max_pages = max(1, min(int(max_pages), MAX_PAGES_CAP))
    threads = max(1, min(int(threads), MAX_THREADS_CAP))
    delay_seconds = max(0.0, min(float(delay_seconds), 10.0))

    with _crawl_lock:
        running = store.crawl_running()
        if running and _active["crawl_id"] == running["id"]:
            return {
                "error": "a crawl is already running",
                "running_crawl_id": running["id"],
                "domain": running["domain"],
                "hint": "wait for it with crawl_status, or stop_crawl(crawl_id) first",
            }
        domain = _domain_key(start_url)
        saved = store.site_headers_get(domain)
        settings = {
            "max_pages": max_pages, "scope": scope, "threads": threads,
            "delay": delay_seconds, "spider_user_agent": bool(spider_user_agent),
            "site_headers_used": bool(saved),
        }
        crawl_id = store.crawl_create(domain, start_url, settings)
        stop = threading.Event()
        _active["crawl_id"] = crawl_id
        _active["stop"] = stop
    run_settings = {**settings, "_headers": (saved or {}).get("headers")}
    threading.Thread(target=_run_crawl, args=(crawl_id, start_url, run_settings, stop), daemon=True).start()
    est = max_pages * (delay_seconds + 1.0) / threads
    return {
        "crawl_id": crawl_id, "domain": domain, "status": "running",
        "settings": settings,
        "estimated_seconds": int(est),
        "site_headers_used": bool(saved),
        "next": "call crawl_status(crawl_id) until status == 'done'",
    }


@mcp.tool()
def crawl_status(crawl_id: int) -> dict:
    """Progress of a crawl: status (running/done/stopped/failed), pages crawled so far, last message."""
    c = _crawl_or_error(crawl_id)
    return {
        "crawl_id": c["id"], "domain": c["domain"], "status": c["status"],
        "crawled": c["crawled"], "max_pages": c["settings"].get("max_pages"),
        "progress": c.get("progress"), "pages": c["pages"], "images": c["images"],
        "blocked": c["blocked"], "error": c.get("error"),
        "started_at": c["started_at"], "finished_at": c.get("finished_at"),
    }


@mcp.tool()
def stop_crawl(crawl_id: int) -> dict:
    """Stop a running crawl early. Pages crawled so far are kept and the crawl is marked 'stopped'."""
    with _crawl_lock:
        if _active["crawl_id"] != crawl_id or _active["stop"] is None:
            c = _crawl_or_error(crawl_id)
            return {"crawl_id": crawl_id, "status": c["status"], "note": "not running"}
        _active["stop"].set()
    return {"crawl_id": crawl_id, "status": "stopping"}


@mcp.tool()
def list_crawls(domain: str | None = None, limit: int = 20) -> list[dict]:
    """Past crawls, newest first. Filter by domain (e.g. 'malebasics.com') to find crawls to compare."""
    rows = store.crawl_list(_domain_key(domain) if domain else None, max(1, min(int(limit), 100)))
    return [
        {"crawl_id": c["id"], "domain": c["domain"], "status": c["status"], "pages": c["pages"],
         "blocked": c["blocked"], "started_at": c["started_at"], "finished_at": c.get("finished_at"),
         "scope": c["settings"].get("scope")}
        for c in rows
    ]


@mcp.tool()
def delete_crawl(crawl_id: int) -> dict:
    """Delete a stored crawl and all its pages."""
    _crawl_or_error(crawl_id)
    return {"crawl_id": crawl_id, "deleted": store.crawl_delete(crawl_id)}


# ---------- tools: reading results ----------

@mcp.tool()
def crawl_summary(crawl_id: int) -> dict:
    """Overview of a finished crawl: status code counts, indexability, and counts for every issue type.
    Use crawl_issues(crawl_id, issue) to list the affected URLs."""
    c = _crawl_or_error(crawl_id)
    pages = store.pages(crawl_id)
    html = [p for p in pages if "text/html" in str(p.get("Content Type") or "")]
    status_counts = Counter(str(p.get("Status Code")) for p in pages)
    issues = {k: len(v) for k, v in _issue_index(crawl_id, html, pages).items()}
    img_counts = Counter(i["alt_status"] for i in store.images(crawl_id))
    return {
        "crawl_id": crawl_id, "domain": c["domain"], "status": c["status"],
        "started_at": c["started_at"], "finished_at": c.get("finished_at"),
        "total_urls": len(pages), "html_pages": len(html),
        "status_codes": dict(sorted(status_counts.items())),
        "indexable": sum(1 for p in html if p.get("Indexability") == "Indexable"),
        "non_indexable": sum(1 for p in html if p.get("Indexability") == "Non-Indexable"),
        "blocked": c["blocked"],
        "images": {"total": sum(img_counts.values()), "missing_alt": img_counts.get("Missing", 0),
                   "empty_alt": img_counts.get("Empty", 0)},
        "issues": issues,
        "avg_response_time_s": round(sum(float(p.get("Response Time") or 0) for p in html) / len(html), 2) if html else None,
    }


ISSUE_TYPES = [
    "broken_4xx", "server_5xx", "blocked_403_429", "redirects", "missing_title", "duplicate_title",
    "long_title", "short_title", "missing_meta_description", "duplicate_meta_description",
    "long_meta_description", "missing_h1", "multiple_h1", "non_indexable", "canonicalised",
    "thin_content", "slow_pages", "orphan_pages", "deep_pages", "missing_lang",
]


def _issue_index(crawl_id: int, html: list[dict], pages: list[dict]) -> dict[str, list[dict]]:
    by_title: dict[str, list[dict]] = defaultdict(list)
    by_desc: dict[str, list[dict]] = defaultdict(list)
    for p in html:
        t = (p.get("Title 1") or "").strip()
        d = (p.get("Meta Description 1") or "").strip()
        if t:
            by_title[t].append(p)
        if d:
            by_desc[d].append(p)
    code = lambda p: int(p.get("Status Code") or 0)  # noqa: E731
    out: dict[str, list[dict]] = {
        "broken_4xx": [_brief(p) for p in pages if 400 <= code(p) < 500 and code(p) not in (403, 429)],
        "server_5xx": [_brief(p) for p in pages if 500 <= code(p) < 600],
        "blocked_403_429": [_brief(p) for p in pages if code(p) in (403, 429)],
        "redirects": [_brief(p, "Redirect URL", "Redirect Type") for p in pages if (p.get("Redirect URL") or "").strip()],
        "missing_title": [_brief(p, "H1-1") for p in html if not (p.get("Title 1") or "").strip()],
        "duplicate_title": [_brief(p) for t, ps in by_title.items() if len(ps) > 1 for p in ps],
        "long_title": [_brief(p, "Title 1 Length") for p in html if int(p.get("Title 1 Length") or 0) > 60],
        "short_title": [_brief(p, "Title 1 Length") for p in html if 0 < int(p.get("Title 1 Length") or 0) < 30],
        "missing_meta_description": [_brief(p) for p in html if not (p.get("Meta Description 1") or "").strip()],
        "duplicate_meta_description": [_brief(p, "Meta Description 1") for d, ps in by_desc.items() if len(ps) > 1 for p in ps],
        "long_meta_description": [_brief(p, "Meta Description 1 Length") for p in html if int(p.get("Meta Description 1 Length") or 0) > 160],
        "missing_h1": [_brief(p) for p in html if not (p.get("H1-1") or "").strip()],
        "multiple_h1": [_brief(p, "H1-1", "H1-2") for p in html if (p.get("H1-2") or "").strip()],
        "non_indexable": [_brief(p, "Indexability Status", "Meta Robots 1") for p in html if p.get("Indexability") == "Non-Indexable"],
        "canonicalised": [_brief(p, "Canonical Link Element 1") for p in html if p.get("Indexability Status") == "Canonicalised"],
        "thin_content": [_brief(p, "Word Count") for p in html if code(p) == 200 and int(p.get("Word Count") or 0) < 100],
        "slow_pages": [_brief(p, "Response Time") for p in html if float(p.get("Response Time") or 0) > 2.0],
        "orphan_pages": [_brief(p, "Crawl Depth") for p in html if int(p.get("Inlinks") or 0) == 0 and int(p.get("Crawl Depth") or 0) > 0],
        "deep_pages": [_brief(p, "Crawl Depth") for p in html if int(p.get("Crawl Depth") or 0) >= 4],
        "missing_lang": [_brief(p) for p in html if code(p) == 200 and not (p.get("Language") or "").strip()],
    }
    return out


@mcp.tool()
def crawl_issues(crawl_id: int, issue: str, limit: int = 100, offset: int = 0) -> dict:
    """URLs affected by one issue type. issue is one of: broken_4xx, server_5xx, blocked_403_429,
    redirects, missing_title, duplicate_title, long_title, short_title, missing_meta_description,
    duplicate_meta_description, long_meta_description, missing_h1, multiple_h1, non_indexable,
    canonicalised, thin_content, slow_pages, orphan_pages (found only via sitemap, no internal links),
    deep_pages (4+ clicks from home), missing_lang."""
    if issue not in ISSUE_TYPES:
        raise ValueError(f"unknown issue '{issue}'. One of: {', '.join(ISSUE_TYPES)}")
    _crawl_or_error(crawl_id)
    pages = store.pages(crawl_id)
    html = [p for p in pages if "text/html" in str(p.get("Content Type") or "")]
    rows = _issue_index(crawl_id, html, pages)[issue]
    limit = max(1, min(int(limit), 500))
    return {"crawl_id": crawl_id, "issue": issue, "total": len(rows),
            "offset": offset, "items": rows[offset:offset + limit]}


@mcp.tool()
def list_pages(crawl_id: int, status_code: int | None = None, url_contains: str | None = None,
               limit: int = 100, offset: int = 0) -> dict:
    """Pages from a crawl with their main SEO fields. Filter by status_code and/or a substring of the URL."""
    _crawl_or_error(crawl_id)
    pages = store.pages(crawl_id)
    if status_code is not None:
        pages = [p for p in pages if int(p.get("Status Code") or 0) == int(status_code)]
    if url_contains:
        needle = url_contains.lower()
        pages = [p for p in pages if needle in str(p.get("Address") or "").lower()]
    limit = max(1, min(int(limit), 500))
    items = [
        {"url": p.get("Address"), "status": p.get("Status Code"), "indexability": p.get("Indexability"),
         "title": p.get("Title 1"), "title_length": p.get("Title 1 Length"),
         "meta_description": p.get("Meta Description 1"), "h1": p.get("H1-1"),
         "canonical": p.get("Canonical Link Element 1"), "word_count": p.get("Word Count"),
         "inlinks": p.get("Inlinks"), "depth": p.get("Crawl Depth"), "response_time_s": p.get("Response Time")}
        for p in pages[offset:offset + limit]
    ]
    return {"crawl_id": crawl_id, "total": len(pages), "offset": offset, "items": items}


@mcp.tool()
def get_page(crawl_id: int, url: str) -> dict:
    """Every data point captured for one URL in a crawl (50+ fields, Screaming Frog style)."""
    _crawl_or_error(crawl_id)
    p = store.page(crawl_id, url.strip())
    if not p:
        raise ValueError("URL not found in this crawl (URLs are stored normalised: no trailing slash)")
    return p


@mcp.tool()
def images_missing_alt(crawl_id: int, limit: int = 200, offset: int = 0) -> dict:
    """Images with missing or empty alt text, with the page each one is on."""
    _crawl_or_error(crawl_id)
    rows = [i for i in store.images(crawl_id) if i["alt_status"] in ("Missing", "Empty")]
    limit = max(1, min(int(limit), 1000))
    return {"crawl_id": crawl_id, "total": len(rows), "offset": offset, "items": rows[offset:offset + limit]}


@mcp.tool()
def compare_crawls(old_crawl_id: int, new_crawl_id: int) -> dict:
    """What changed between two crawls of the same domain: new and removed URLs, status code changes,
    and title / meta description / H1 changes. Use list_crawls(domain) to pick the ids."""
    old = _crawl_or_error(old_crawl_id)
    new = _crawl_or_error(new_crawl_id)
    if old["domain"] != new["domain"]:
        raise ValueError(f"crawls are for different domains ({old['domain']} vs {new['domain']})")
    op = {p["Address"]: p for p in store.pages(old_crawl_id)}
    np_ = {p["Address"]: p for p in store.pages(new_crawl_id)}
    added = sorted(set(np_) - set(op))
    removed = sorted(set(op) - set(np_))
    status_changes, content_changes = [], []
    for url in sorted(set(op) & set(np_)):
        a, b = op[url], np_[url]
        if a.get("Status Code") != b.get("Status Code"):
            status_changes.append({"url": url, "old": a.get("Status Code"), "new": b.get("Status Code")})
        diff = {}
        for field, key in (("title", "Title 1"), ("meta_description", "Meta Description 1"), ("h1", "H1-1"), ("canonical", "Canonical Link Element 1")):
            if (a.get(key) or "") != (b.get(key) or ""):
                diff[field] = {"old": a.get(key), "new": b.get(key)}
        if diff:
            content_changes.append({"url": url, **diff})
    return {
        "domain": new["domain"],
        "old": {"crawl_id": old_crawl_id, "finished_at": old.get("finished_at"), "pages": old["pages"]},
        "new": {"crawl_id": new_crawl_id, "finished_at": new.get("finished_at"), "pages": new["pages"]},
        "added_urls": {"count": len(added), "items": added[:200]},
        "removed_urls": {"count": len(removed), "items": removed[:200]},
        "status_changes": {"count": len(status_changes), "items": status_changes[:200]},
        "content_changes": {"count": len(content_changes), "items": content_changes[:200]},
    }


# ---------- tools: per-site request headers (Shopify crawler signatures) ----------

@mcp.tool()
def set_site_headers(domain: str, headers: str, note: str | None = None, expires_at: str | None = None) -> dict:
    """Save request headers to send on every crawl of a domain. Needed for Shopify stores: in the store
    admin go to Online Store → Preferences → Crawler access → Create signature, then pass the three
    headers it shows, one per line, as 'Name: value' (Signature-Input, Signature, Signature-Agent).
    Signatures last at most 3 months; pass expires_at (YYYY-MM-DD) so it can be tracked."""
    parsed = parse_header_lines(headers)
    if not parsed:
        raise ValueError("no 'Name: value' lines found in headers")
    d = _domain_key(domain)
    store.site_headers_set(d, parsed, note, expires_at)
    return {"domain": d, "headers": sorted(parsed), "expires_at": expires_at}


@mcp.tool()
def list_site_headers() -> list[dict]:
    """Domains that have saved request headers (values are not returned), with their expiry."""
    today = datetime.now(timezone.utc).date().isoformat()
    out = []
    for r in store.site_headers_list():
        out.append({**r, "expired": bool(r.get("expires_at") and r["expires_at"] < today)})
    return out


@mcp.tool()
def delete_site_headers(domain: str) -> dict:
    """Remove the saved request headers for a domain."""
    d = _domain_key(domain)
    return {"domain": d, "deleted": store.site_headers_delete(d)}


# ---------- ASGI app ----------

class HealthCheckMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path == "/healthz":
            running = store.crawl_running()
            return JSONResponse({"ok": True, "running_crawl": running["id"] if running else None})
        return await call_next(request)


class AcceptHeaderShimMiddleware:
    """The MCP SDK rejects POSTs whose Accept header lacks either media type;
    some clients send only one. Rewrite it on /mcp so they still work."""

    _REQUIRED = b"application/json, text/event-stream"

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            if path == "/mcp" or path.startswith("/mcp/"):
                headers = scope.get("headers", [])
                accept = next((v for k, v in headers if k == b"accept"), b"").lower()
                if not accept or accept.strip() == b"*/*" or b"application/json" not in accept or b"text/event-stream" not in accept:
                    scope = dict(scope)
                    scope["headers"] = [(k, v) for k, v in headers if k != b"accept"] + [(b"accept", self._REQUIRED)]
        await self.app(scope, receive, send)


class BearerAuthMiddleware:
    """Static MCP_BEARER_TOKEN (Bearer or X-API-Key) or an OAuth access token on everything but OPEN_PATHS."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "") if scope.get("type") == "http" else ""
        if scope.get("type") != "http" or path in oauth_server.OPEN_PATHS:
            await self.app(scope, receive, send)
            return
        auth = api_key = ""
        for k, v in scope.get("headers", []):
            if k == b"authorization":
                auth = v.decode("latin1")
            elif k == b"x-api-key":
                api_key = v.decode("latin1")
        token = auth[7:].strip() if auth[:7].lower() == "bearer " else api_key.strip()
        if oauth_server.validate_bearer_token(token):
            await self.app(scope, receive, send)
            return
        body = b'{"error": "unauthorized"}'
        www = f'Bearer resource_metadata="{oauth_server.PUBLIC_BASE_URL}/.well-known/oauth-protected-resource"'.encode()
        await send({"type": "http.response.start", "status": 401, "headers": [
            (b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()), (b"www-authenticate", www)]})
        await send({"type": "http.response.body", "body": body})


async def _robots(_request):
    from starlette.responses import PlainTextResponse
    return PlainTextResponse("User-agent: *\nDisallow: /\n")


def build_app() -> Starlette:
    oauth_store.init_schema()
    oauth_store.purge_expired()
    store.init_schema()
    # A crawl that was running when the container last stopped can never finish.
    stale = store.crawl_running()
    if stale:
        store.crawl_fail(stale["id"], "server restarted while crawling")
    inner = mcp.streamable_http_app()
    return Starlette(
        routes=[Route("/robots.txt", _robots), *oauth_server.oauth_routes(), Mount("/", app=AcceptHeaderShimMiddleware(inner))],
        middleware=[Middleware(HealthCheckMiddleware), Middleware(BearerAuthMiddleware)],
        lifespan=inner.router.lifespan_context,
    )


app = build_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")), proxy_headers=True, forwarded_allow_ips="*")

"""OAuth 2.1 authorization server for clients connecting TO this MCP service.

Ported from klaviyo-mcp on this VPS (itself ported from ebay-platform), with
the same shape because it is known to work with real Claude and ChatGPT
clients: discovery metadata, Dynamic Client Registration (RFC 7591),
authorization-code flow with PKCE (S256), refresh tokens, and bearer
validation that also accepts the static MCP_BEARER_TOKEN for header clients.

Consent is approved by entering the staff access key (STAFF_ACCESS_KEY) or
the static MCP_BEARER_TOKEN.
"""

from __future__ import annotations

import base64
import hashlib
import html
import os
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone

import oauth_store as store

PUBLIC_BASE_URL = os.environ.get("PUBLIC_BASE_URL", "https://seo-mcp.malebasics.com").rstrip("/")

ACCESS_TOKEN_TTL_SECONDS = 3600
REFRESH_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 30
AUTH_CODE_TTL_SECONDS = 60 * 10


def _consent_secrets() -> list[str]:
    return [s for s in (os.environ.get("MCP_BEARER_TOKEN", ""), os.environ.get("STAFF_ACCESS_KEY", "")) if s]


def authorization_server_metadata() -> dict:
    return {
        "issuer": PUBLIC_BASE_URL,
        "authorization_endpoint": f"{PUBLIC_BASE_URL}/oauth/authorize",
        "token_endpoint": f"{PUBLIC_BASE_URL}/oauth/token",
        "registration_endpoint": f"{PUBLIC_BASE_URL}/oauth/register",
        "scopes_supported": ["mcp"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "token_endpoint_auth_methods_supported": ["client_secret_post", "client_secret_basic", "none"],
        "code_challenge_methods_supported": ["S256"],
    }


def protected_resource_metadata() -> dict:
    return {
        "resource": f"{PUBLIC_BASE_URL}/mcp",
        "authorization_servers": [PUBLIC_BASE_URL],
        "scopes_supported": ["mcp"],
        "bearer_methods_supported": ["header"],
    }


def register_client(body: dict) -> dict:
    redirect_uris = body.get("redirect_uris") or []
    if not isinstance(redirect_uris, list) or not redirect_uris:
        raise ValueError("redirect_uris is required and must be a non-empty array")
    client_name = body.get("client_name") or "Unknown Client"
    grant_types = body.get("grant_types") or ["authorization_code", "refresh_token"]
    response_types = body.get("response_types") or ["code"]
    token_endpoint_auth_method = body.get("token_endpoint_auth_method") or "client_secret_post"
    scope = body.get("scope") or "mcp"

    client_id = "mcp_" + secrets.token_urlsafe(16)
    client_secret = None if token_endpoint_auth_method == "none" else secrets.token_urlsafe(32)

    store.oauth_client_create(
        client_id=client_id,
        client_secret=client_secret,
        client_name=client_name,
        redirect_uris=redirect_uris,
        grant_types=grant_types,
        response_types=response_types,
        token_endpoint_auth_method=token_endpoint_auth_method,
        scope=scope,
    )
    response = {
        "client_id": client_id,
        "client_id_issued_at": int(datetime.now(timezone.utc).timestamp()),
        "client_name": client_name,
        "redirect_uris": redirect_uris,
        "grant_types": grant_types,
        "response_types": response_types,
        "token_endpoint_auth_method": token_endpoint_auth_method,
        "scope": scope,
    }
    if client_secret:
        response["client_secret"] = client_secret
        response["client_secret_expires_at"] = 0
    return response


def verify_pkce(code_verifier: str, code_challenge: str, method: str | None) -> bool:
    if not code_challenge:
        return True
    if (method or "S256") != "S256":
        return False
    digest = hashlib.sha256((code_verifier or "").encode("utf-8")).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return secrets.compare_digest(expected, code_challenge)


def issue_auth_code(client_id, redirect_uri, scope, code_challenge, code_challenge_method) -> str:
    code = secrets.token_urlsafe(32)
    store.oauth_auth_code_create(
        code=code,
        client_id=client_id,
        redirect_uri=redirect_uri,
        scope=scope,
        code_challenge=code_challenge,
        code_challenge_method=code_challenge_method,
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=AUTH_CODE_TTL_SECONDS),
    )
    return code


def issue_tokens(client_id: str, scope: str) -> dict:
    access_token = secrets.token_urlsafe(48)
    refresh_token = secrets.token_urlsafe(48)
    now = datetime.now(timezone.utc)
    store.oauth_access_token_create(
        token=access_token,
        client_id=client_id,
        scope=scope,
        expires_at=now + timedelta(seconds=ACCESS_TOKEN_TTL_SECONDS),
        refresh_token=refresh_token,
        refresh_expires_at=now + timedelta(seconds=REFRESH_TOKEN_TTL_SECONDS),
    )
    return {
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": ACCESS_TOKEN_TTL_SECONDS,
        "refresh_token": refresh_token,
        "scope": scope,
    }


def validate_bearer_token(token: str) -> bool:
    if not token:
        return False
    static = os.environ.get("MCP_BEARER_TOKEN", "")
    if static and secrets.compare_digest(token, static):
        return True
    return store.oauth_access_token_validate(token) is not None


def authenticate_client(client_id_form, client_secret_form, auth_header) -> dict | None:
    client_id = client_id_form
    client_secret = client_secret_form
    if not client_id and auth_header and auth_header.lower().startswith("basic "):
        try:
            raw = base64.b64decode(auth_header[6:].strip()).decode("utf-8")
            client_id, _, client_secret = raw.partition(":")
        except Exception:
            return None
    if not client_id:
        return None
    record = store.oauth_client_get(client_id)
    if not record:
        return None
    if record.get("token_endpoint_auth_method") == "none":
        return record
    expected = record.get("client_secret")
    if not expected or not client_secret:
        return None
    if not secrets.compare_digest(expected, client_secret):
        return None
    return record


# ---------------------------------------------------------------------------
# Starlette routes (form bodies parsed manually to avoid extra dependencies)
# ---------------------------------------------------------------------------

from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse  # noqa: E402
from starlette.routing import Route  # noqa: E402


async def _form(request) -> dict:
    body = (await request.body()).decode("utf-8", "replace")
    return {k: v[0] for k, v in urllib.parse.parse_qs(body).items()}


async def _ep_as_metadata(request):
    return JSONResponse(authorization_server_metadata())


async def _ep_pr_metadata(request):
    return JSONResponse(protected_resource_metadata())


async def _ep_register(request):
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "invalid_client_metadata"}, status_code=400)
    try:
        return JSONResponse(register_client(body or {}), status_code=201)
    except ValueError as exc:
        return JSONResponse({"error": "invalid_client_metadata", "error_description": str(exc)}, status_code=400)


async def _ep_authorize(request):
    q = request.query_params
    client = store.oauth_client_get(q.get("client_id", ""))
    if not client:
        return HTMLResponse("<h1>Unknown client_id</h1>", status_code=400)
    if q.get("redirect_uri", "") not in client["redirect_uris"]:
        return HTMLResponse("<h1>redirect_uri not registered for this client</h1>", status_code=400)
    h = lambda v: html.escape(v or "", quote=True)  # noqa: E731
    return HTMLResponse(f"""<!doctype html><html><head><meta name="robots" content="noindex"><title>Approve access</title></head>
<body style="font-family:sans-serif;max-width:480px;margin:60px auto;padding:0 16px">
<h2>SEO Spider MCP — approve access?</h2>
<p><b>{h(client.get('client_name'))}</b> is requesting access to the MaleBasics marketing data server.</p>
<p>To approve, enter the <b>staff access key</b>:</p>
<form method="POST" action="/oauth/authorize/consent">
  <input type="hidden" name="client_id" value="{h(q.get('client_id'))}">
  <input type="hidden" name="redirect_uri" value="{h(q.get('redirect_uri'))}">
  <input type="hidden" name="state" value="{h(q.get('state'))}">
  <input type="hidden" name="scope" value="{h(q.get('scope') or 'mcp')}">
  <input type="hidden" name="code_challenge" value="{h(q.get('code_challenge'))}">
  <input type="hidden" name="code_challenge_method" value="{h(q.get('code_challenge_method'))}">
  <input type="password" name="master_token" required autofocus
         style="width:100%;padding:10px;margin-bottom:16px;border:1px solid #ccc;border-radius:4px">
  <div style="display:flex;gap:12px">
    <button type="submit" name="decision" value="approve"
            style="flex:1;padding:10px;background:#111;color:#fff;border:0;border-radius:4px;font-weight:bold;cursor:pointer">Approve</button>
    <button type="submit" name="decision" value="deny"
            style="flex:1;padding:10px;background:#eee;color:#333;border:0;border-radius:4px;cursor:pointer">Deny</button>
  </div>
</form></body></html>""")


async def _ep_consent(request):
    form = await _form(request)
    redirect_uri = form.get("redirect_uri", "")
    state = form.get("state")

    def _redirect(params: str):
        sep = "&" if "?" in redirect_uri else "?"
        loc = f"{redirect_uri}{sep}{params}"
        if state:
            loc += f"&state={urllib.parse.quote(state)}"
        return RedirectResponse(url=loc, status_code=302)

    if form.get("decision") != "approve":
        return _redirect("error=access_denied")
    master = form.get("master_token", "")
    if not any(secrets.compare_digest(master, s) for s in _consent_secrets()):
        return HTMLResponse("<h1>Wrong key.</h1><p><a href='javascript:history.back()'>Back</a></p>", status_code=403)
    client = store.oauth_client_get(form.get("client_id", ""))
    if not client or redirect_uri not in client["redirect_uris"]:
        return HTMLResponse("<h1>Invalid client or redirect_uri</h1>", status_code=400)
    code = issue_auth_code(
        client_id=form["client_id"],
        redirect_uri=redirect_uri,
        scope=form.get("scope") or "mcp",
        code_challenge=form.get("code_challenge") or None,
        code_challenge_method=form.get("code_challenge_method") or None,
    )
    return _redirect(f"code={urllib.parse.quote(code)}")


async def _ep_token(request):
    form = await _form(request)
    grant_type = form.get("grant_type", "")
    client = authenticate_client(form.get("client_id"), form.get("client_secret"), request.headers.get("authorization"))
    if not client:
        return JSONResponse({"error": "invalid_client"}, status_code=401)

    if grant_type == "authorization_code":
        record = store.oauth_auth_code_consume(form.get("code", ""))
        if not record:
            return JSONResponse({"error": "invalid_grant", "error_description": "Code missing, expired, or already used"}, status_code=400)
        if record["client_id"] != client["client_id"]:
            return JSONResponse({"error": "invalid_grant", "error_description": "Code was issued to a different client"}, status_code=400)
        if record["redirect_uri"] != form.get("redirect_uri", ""):
            return JSONResponse({"error": "invalid_grant", "error_description": "redirect_uri mismatch"}, status_code=400)
        if record.get("code_challenge") and not verify_pkce(form.get("code_verifier", ""), record["code_challenge"], record.get("code_challenge_method")):
            return JSONResponse({"error": "invalid_grant", "error_description": "PKCE verification failed"}, status_code=400)
        return JSONResponse(issue_tokens(client["client_id"], record["scope"] or "mcp"))

    if grant_type == "refresh_token":
        record = store.oauth_refresh_token_lookup(form.get("refresh_token", ""))
        if not record or record["client_id"] != client["client_id"]:
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        expires = record.get("refresh_expires_at")
        if expires and expires < datetime.now(timezone.utc):
            return JSONResponse({"error": "invalid_grant", "error_description": "Refresh token expired"}, status_code=400)
        return JSONResponse(issue_tokens(client["client_id"], record["scope"] or "mcp"))

    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


OPEN_PATHS = {
    "/healthz",
    "/robots.txt",
    "/.well-known/oauth-authorization-server",
    "/.well-known/oauth-authorization-server/mcp",
    "/.well-known/oauth-protected-resource",
    "/.well-known/oauth-protected-resource/mcp",
    "/oauth/register",
    "/oauth/authorize",
    "/oauth/authorize/consent",
    "/oauth/token",
}


def oauth_routes() -> list:
    return [
        Route("/.well-known/oauth-authorization-server", _ep_as_metadata, methods=["GET"]),
        Route("/.well-known/oauth-authorization-server/mcp", _ep_as_metadata, methods=["GET"]),
        Route("/.well-known/oauth-protected-resource", _ep_pr_metadata, methods=["GET"]),
        Route("/.well-known/oauth-protected-resource/mcp", _ep_pr_metadata, methods=["GET"]),
        Route("/oauth/register", _ep_register, methods=["POST"]),
        Route("/oauth/authorize", _ep_authorize, methods=["GET"]),
        Route("/oauth/authorize/consent", _ep_consent, methods=["POST"]),
        Route("/oauth/token", _ep_token, methods=["POST"]),
    ]

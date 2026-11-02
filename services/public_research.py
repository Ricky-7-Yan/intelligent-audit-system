"""Public-source research through anonymous Parallel Search MCP.

Sending a query to a third-party provider moves data outside the deployment, so
it is gated three times before a single byte leaves: ``SEARCH_PROVIDER`` must be
enabled by an administrator, the caller must hold
``execute:external_research``, and the query must clear the same credential, PII
and confidential-business screening the upload path applies. Every attempt is
recorded as an audit event that carries a query fingerprint, never the text.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx

from config import SEARCH_CONFIG
from services.safety_gate import SafetyGate
from services.security import (
    EXTERNAL_RESEARCH_PERMISSION,
    audit_event_store,
    current_principal,
    has_permission,
)
from services.upload_security import scan_sensitive_text

# The MCP client SDK is an optional extra; see requirements-search.txt. The
# default install stays free of it so that a deployment which never enables
# public search does not carry the SDK or its transitive dependencies.
try:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
except ImportError:  # pragma: no cover - exercised by patching the sentinels
    ClientSession = None
    streamable_http_client = None


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "AuditPilot/4.2.0 (+https://github.com/Ricky-7-Yan/intelligent-audit-system)"
SDK_REQUIRED = "Public search needs the optional MCP client; install requirements-search.txt or set SEARCH_PROVIDER=disabled."

_safety_gate = SafetyGate()


def public_research(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve the provider without deriving an external query from private context."""
    query = payload.get("public_research_query")
    if not query:
        return {"status": "not_requested", "sources": []}
    configured = SEARCH_CONFIG["provider"]
    provider = payload.get("search_provider", configured)
    if configured == "disabled" or provider == "disabled":
        return {"status": "disabled", "sources": []}
    if provider != "parallel" or configured not in {"parallel", "disabled"}:
        raise ValueError("Unsupported search provider; choose parallel or disabled.")
    if not isinstance(query, str) or not query.strip() or len(query) > 1000:
        raise ValueError("public_research_query must contain 1 to 1000 public characters.")
    session_id = payload.get("public_research_session_id", "")
    if not isinstance(session_id, str) or not re.fullmatch(r"[a-f0-9]{32,64}", session_id):
        session_id = uuid.uuid4().hex
    fingerprint = {
        "provider": "parallel",
        "destination": MCP_URL,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "query_chars": len(query),
        "session_id": session_id,
    }

    _authorize(fingerprint)
    _screen(query, fingerprint)
    if ClientSession is None or streamable_http_client is None:
        raise RuntimeError(SDK_REQUIRED)

    audit_event_store().append(
        "external_research_dispatched", MCP_URL, 202, {**fingerprint, "screening": "passed"}
    )
    try:
        sources = anyio.run(search, query, session_id)
    except Exception as exc:
        audit_event_store().append(
            "external_research_failed", MCP_URL, 502, {**fingerprint, "error_type": exc.__class__.__name__}
        )
        # Provider responses may contain query text. Keep the user-facing error bounded.
        raise RuntimeError("Public web search failed; retry later or disable search and review sources manually.") from exc
    audit_event_store().append(
        "external_research_completed", MCP_URL, 200, {**fingerprint, "source_count": len(sources)}
    )
    return {
        "status": "completed",
        "provider": "parallel",
        "sources": sources,
        "review_required": True,
        "notice": "Public references are supplementary, untrusted material, not enterprise control evidence.",
    }


def _authorize(fingerprint: dict[str, Any]) -> None:
    """Require a transfer permission that no read-only knowledge role carries."""

    principal = current_principal()
    if has_permission(principal, EXTERNAL_RESEARCH_PERMISSION):
        return
    audit_event_store().append(
        "external_research_denied",
        MCP_URL,
        403,
        {**fingerprint, "required_permission": EXTERNAL_RESEARCH_PERMISSION},
    )
    raise PermissionError(f"缺少权限：{EXTERNAL_RESEARCH_PERMISSION}")


def _screen(query: str, fingerprint: dict[str, Any]) -> None:
    """Apply the upload screening vocabulary and the runtime gate to the query."""

    categories = sorted(category for category, hits in scan_sensitive_text(query).items() if hits)
    gate = _safety_gate.inspect({"public_research_query": query}, stage="external_research")
    if gate["status"] == "blocked":
        categories = sorted({*categories, *(finding["code"] for finding in gate["findings"])})
    if not categories:
        return
    audit_event_store().append(
        "external_research_blocked", MCP_URL, 422, {**fingerprint, "blocked_categories": categories}
    )
    raise ValueError(
        "public_research_query 疑似包含凭据、个人信息或内部商业信息，请脱敏后重试：" + ", ".join(categories)
    )


async def search(query: str, session_id: str) -> list[dict[str, Any]]:
    """Bound the whole MCP exchange; cancellation closes both session and transport."""
    timeout = min(6.0, max(0.01, float(SEARCH_CONFIG["timeout_seconds"])))
    with anyio.fail_after(timeout):
        async with httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=timeout, follow_redirects=False,
        ) as client:
            async with streamable_http_client(MCP_URL, http_client=client) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools = await session.list_tools()
                    if "web_search" not in {tool.name for tool in tools.tools}:
                        raise ValueError("Search MCP does not advertise web_search.")
                    result = await session.call_tool("web_search", {
                        "objective": query,
                        "search_queries": [query],
                        "session_id": session_id,
                    })
                    if result.isError:
                        raise ValueError("Search MCP returned a tool error.")
                    data = result.structuredContent
                    if data is None:
                        text = "".join(block.text for block in result.content if block.type == "text")
                        data = json.loads(text)
                    return _sources(data)


def _sources(data: Any) -> list[dict[str, Any]]:
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError("Search MCP returned an invalid result document.")
    sources = []
    seen = set()
    for item in data["results"]:
        if not isinstance(item, dict) or not isinstance(item.get("url"), str):
            continue
        url = item["url"]
        try:
            parsed = urlsplit(url)
            usable = parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password
        except ValueError:
            usable = False
        if not usable or url in seen:
            continue
        seen.add(url)
        excerpts = item.get("excerpts")
        sources.append({
            "url": url,
            "title": str(item.get("title") or url)[:300],
            "excerpts": [text[:2000] for text in excerpts[:3] if isinstance(text, str)] if isinstance(excerpts, list) else [],
            "publish_date": item.get("publish_date") if isinstance(item.get("publish_date"), str) else None,
            "evidence_type": "public_reference",
        })
        if len(sources) == 8:
            break
    return sources

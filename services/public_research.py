"""Public-source research through anonymous Parallel Search MCP."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any
from urllib.parse import urlsplit

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from config import SEARCH_CONFIG


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = "AuditPilot/4.2.0 (+https://github.com/Ricky-7-Yan/intelligent-audit-system)"


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
    try:
        sources = anyio.run(search, query, session_id)
    except Exception as exc:
        # Provider responses may contain query text. Keep the user-facing error bounded.
        raise RuntimeError("Public web search failed; retry later or disable search and review sources manually.") from exc
    return {
        "status": "completed",
        "provider": "parallel",
        "sources": sources,
        "review_required": True,
        "notice": "Public references are supplementary, untrusted material, not enterprise control evidence.",
    }


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

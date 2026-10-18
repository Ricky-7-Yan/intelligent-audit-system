from __future__ import annotations

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from config import SEARCH_CONFIG
from services.agent_runtime import AgentRuntime
from services.public_research import MCP_URL, USER_AGENT, search
from services.skill_registry import SkillRegistry


REAL_CLIENT = httpx.AsyncClient
PUBLIC_QUERY = "NIST access control guidance"
SOURCE = {
    "url": "https://www.nist.gov/privacy-framework",
    "title": "NIST Privacy Framework",
    "excerpts": ["Identify and manage privacy risk."],
    "publish_date": None,
}


class MCPFixture:
    """Exercise the real SDK's HTTP handshake, discovery and tool calls."""

    def __init__(self, mode="normal"):
        self.mode = mode
        self.requests = []
        self.cancelled = False
        self.closed = False

    async def handle(self, request):
        self.requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        message = json.loads(request.content)
        method = message["method"]
        if "id" not in message:
            return httpx.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": "2025-11-25",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fixture", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [] if self.mode == "missing" else [{
                "name": "web_search", "inputSchema": {"type": "object"},
            }]}
        elif method == "tools/call":
            if self.mode in {"timeout", "cancel"}:
                try:
                    await asyncio.sleep(60)
                except asyncio.CancelledError:
                    self.cancelled = True
                    raise
            if self.mode == "http_error":
                return httpx.Response(401)
            data = {"results": [SOURCE, SOURCE, {"url": "javascript:alert(1)"}, {"url": "https://name:secret@example.com"}]}
            result = {"content": [{"type": "text", "text": json.dumps(data)}]}
            if self.mode == "structured":
                result["structuredContent"] = data
            elif self.mode == "tool_error":
                result = {"content": [{"type": "text", "text": "provider unavailable"}], "isError": True}
            elif self.mode == "malformed":
                result = {"content": [{"type": "text", "text": "not JSON"}]}
            elif self.mode == "invalid_shape":
                result = {"content": [{"type": "text", "text": "{}"}]}
            elif self.mode == "empty":
                result = {"content": [{"type": "text", "text": '{"results": []}'}]}
        else:
            raise AssertionError(method)
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": message["id"], "result": result})

    def client(self, **kwargs):
        fixture = self

        class Transport(httpx.MockTransport):
            async def aclose(self):
                fixture.closed = True
                await super().aclose()

        return REAL_CLIENT(transport=Transport(self.handle), **kwargs)

    def assert_requests(self, test):
        for request in self.requests:
            test.assertEqual(str(request.url), MCP_URL)
            test.assertEqual(request.headers["User-Agent"], USER_AGENT)
            test.assertNotIn("Authorization", request.headers)
        calls = [json.loads(r.content) for r in self.requests if r.method == "POST"]
        test.assertIn("initialize", [c["method"] for c in calls])
        test.assertIn("tools/list", [c["method"] for c in calls])
        for call in calls:
            if call["method"] == "tools/call":
                test.assertEqual(call["params"]["name"], "web_search")
                args = call["params"]["arguments"]
                test.assertEqual(args["objective"], PUBLIC_QUERY)
                test.assertEqual(args["search_queries"], [PUBLIC_QUERY])
                test.assertEqual(set(args), {"objective", "search_queries", "session_id"})
                test.assertRegex(args["session_id"], r"^[a-f0-9]{32,64}$")
        test.assertTrue(self.closed)


class PublicResearchRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.settings = patch.dict(SEARCH_CONFIG, {"provider": "parallel", "timeout_seconds": 6})
        self.settings.start()
        self.addCleanup(self.settings.stop)
        self.registry = SkillRegistry()
        self.registry.log_file = Path(self.tmp.name) / "runs.jsonl"
        self.addCleanup(self.registry._executor.shutdown, wait=True)

    def execute(self, **extra):
        return self.registry.execute("audit.deep_research_brief", {
            "question": "Private tenant objective must stay local",
            "public_research_query": PUBLIC_QUERY,
            **extra,
        })

    def test_runtime_dispatch_persists_citations_and_only_sends_public_query(self):
        fixture = MCPFixture()
        with patch("services.public_research.httpx.AsyncClient", fixture.client):
            runtime = AgentRuntime(self.registry)
            runtime.runtime_dir = Path(self.tmp.name) / "runtime"
            runtime.runtime_dir.mkdir()
            runtime.event_log = runtime.runtime_dir / "events.jsonl"
            task = runtime.create_task("Prepare private controls", {
                "public_research_query": PUBLIC_QUERY,
                "private_tenant_data": "never transmit this",
            })
            task = runtime.run_until_pause(task["task_id"], max_steps=10)
        artifacts = [a for a in task["artifacts"] if a["type"] == "research"]
        self.assertEqual(len(artifacts), 1)
        research = artifacts[0]["content"]["public_research"]
        self.assertEqual(research["provider"], "parallel")
        self.assertEqual(research["sources"][0]["url"], SOURCE["url"])
        self.assertEqual(research["sources"][0]["excerpts"], SOURCE["excerpts"])
        self.assertEqual(len(research["sources"]), 1)
        self.assertTrue(research["review_required"])
        self.assertIn("query_rewrites", artifacts[0]["content"])
        persisted = runtime.get_task(task["task_id"])
        self.assertEqual(persisted["artifacts"], task["artifacts"])
        fixture.assert_requests(self)

    def test_no_query_and_disabled_choices_never_make_http_requests(self):
        with patch("services.public_research.httpx.AsyncClient", side_effect=AssertionError("unexpected HTTP")):
            result = self.registry.execute("audit.deep_research_brief", {"question": "private research"})
            self.assertEqual(result["output"]["public_research"]["status"], "not_requested")
            self.assertEqual(self.execute(search_provider="disabled")["output"]["public_research"]["status"], "disabled")
            SEARCH_CONFIG["provider"] = "disabled"
            self.assertEqual(self.execute(search_provider="parallel")["output"]["public_research"]["status"], "disabled")

    def test_saved_parallel_and_explicit_override_without_credentials(self):
        fixture = MCPFixture("structured")
        with patch("services.public_research.httpx.AsyncClient", fixture.client):
            self.assertEqual(self.execute(search_provider="parallel")["status"], "success")
        SEARCH_CONFIG["provider"] = "disabled"
        # A prior success must not cache around a newly disabled configuration.
        self.assertEqual(self.execute()["output"]["public_research"]["status"], "disabled")
        fixture.assert_requests(self)

    def test_permissions_and_invalid_inputs_fail_before_network(self):
        with patch("services.public_research.httpx.AsyncClient", side_effect=AssertionError("unexpected HTTP")):
            denied = self.registry.execute("audit.deep_research_brief", {
                "question": "private", "public_research_query": PUBLIC_QUERY,
            }, permission_context=[])
            self.assertEqual(denied["error_type"], "PermissionDeniedError")
            for extra in ({"search_provider": "unknown"}, {"public_research_query": " "}, {"public_research_query": "a" * 1001}):
                self.registry._circuits.clear()
                self.assertEqual(self.execute(**extra)["status"], "failed")

    def test_provider_errors_are_failed_calls_without_fallback(self):
        for mode in ("missing", "http_error", "tool_error", "malformed", "invalid_shape", "timeout"):
            with self.subTest(mode=mode):
                self.registry._circuits.clear()
                fixture = MCPFixture(mode)
                SEARCH_CONFIG["timeout_seconds"] = 0.05 if mode == "timeout" else 6
                with patch("services.public_research.httpx.AsyncClient", fixture.client):
                    result = self.execute()
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error_type"], "RuntimeError")
                self.assertNotIn("public_research", result["output"])
                fixture.assert_requests(self)
                if mode == "timeout":
                    self.assertTrue(fixture.cancelled)

    def test_empty_search_remains_empty_and_requires_review(self):
        fixture = MCPFixture("empty")
        with patch("services.public_research.httpx.AsyncClient", fixture.client):
            result = self.execute()["output"]["public_research"]
        self.assertEqual(result["sources"], [])
        self.assertTrue(result["review_required"])


class PublicResearchCancellationTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_propagates_and_closes_transport(self):
        fixture = MCPFixture("cancel")
        with patch.dict(SEARCH_CONFIG, {"timeout_seconds": 6}), patch(
            "services.public_research.httpx.AsyncClient", fixture.client,
        ):
            task = asyncio.create_task(search(PUBLIC_QUERY, "b" * 32))
            # Wait for the actual tool request rather than cancelling during setup.
            async def wait_for_call():
                while not any(b'"tools/call"' in r.content for r in fixture.requests):
                    await asyncio.sleep(0.005)
            await asyncio.wait_for(wait_for_call(), timeout=2)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(fixture.cancelled)
        fixture.assert_requests(self)

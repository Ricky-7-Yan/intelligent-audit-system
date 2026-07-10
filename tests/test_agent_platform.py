from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.agent_runtime import AgentRuntime
from services.conversation_memory import ConversationMemory
from services.evaluation_repository import EvaluationRunRepository
from services.evolution_harness import EvolutionHarness
from services.intent_router import HybridIntentRouter
from services.safety_gate import SafetyGate
from services.skill_registry import Skill, SkillRegistry


class IntentRouterTests(unittest.TestCase):
    def test_routes_cross_domain_request_to_specialists(self) -> None:
        result = HybridIntentRouter().classify("请分析 ERP 权限日志证据，识别高风险并生成整改计划")
        self.assertIn(result["intent"], {"evidence_analysis", "risk_assessment", "finding_remediation"})
        self.assertGreater(result["confidence"], 0.5)
        self.assertTrue(result["agents"])
        self.assertIn("ERP", " ".join(result["entities"]["systems"]).upper())


class ConversationMemoryTests(unittest.TestCase):
    def test_compacts_working_memory_and_builds_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = ConversationMemory(Path(tmp), compress_at=6, retain_recent=2)
            for index in range(3):
                memory.record_turn(
                    "audit-session",
                    f"第{index}轮：检查 ERP 权限，参考 ISO27001",
                    "已记录证据和控制测试要求。",
                    {"intent": "control_testing", "agents": ["control_agent"]},
                )
            session = memory.get_session("audit-session")
            self.assertIsNotNone(session)
            assert session is not None
            self.assertEqual(len(session["messages"]), 2)
            self.assertEqual(len(session["episodes"]), 1)
            self.assertIn("ISO27001", session["profile"]["standards"])
            context = memory.context_for("audit-session", "继续检查权限证据")
            self.assertIn("会话摘要", context["prompt_text"])


class SkillRegistryTests(unittest.TestCase):
    def test_validation_cache_and_runtime_reflection(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = SkillRegistry()
            registry.log_file = Path(tmp) / "runs.jsonl"
            calls = {"count": 0}

            def handler(payload):
                calls["count"] += 1
                return {"value": payload["value"]}

            registry._register(
                Skill(
                    name="test.cached",
                    title="测试缓存",
                    description="验证输入治理和 TTL 缓存。",
                    input_schema={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                    },
                    permissions=["read:test"],
                    handler=handler,
                    cache_ttl_seconds=60,
                )
            )
            invalid = registry.execute("test.cached", {})
            self.assertEqual(invalid["error_type"], "InputValidationError")
            first = registry.execute("test.cached", {"value": "ok"})
            second = registry.execute("test.cached", {"value": "ok"})
            self.assertEqual(first["status"], "success")
            self.assertTrue(second["cache_hit"])
            self.assertEqual(calls["count"], 1)
            self.assertTrue(registry.delete_run(second["run_id"]))
            self.assertFalse(any(run["run_id"] == second["run_id"] for run in registry.recent_runs(20)))

            runtime = AgentRuntime(registry, SafetyGate())
            runtime.runtime_dir = Path(tmp) / "runtime"
            runtime.runtime_dir.mkdir()
            task = runtime.create_task("生成 ERP 权限审计计划", {"audit_item": "ERP 权限"})
            self.assertTrue(task["reflections"])
            self.assertIn(task["reflections"][0]["verdict"], {"pass", "review"})
            self.assertTrue(runtime.delete_task(task["task_id"]))
            self.assertIsNone(runtime.get_task(task["task_id"]))


class EvaluationRepositoryTests(unittest.TestCase):
    def test_compares_new_run_with_previous_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repository = EvaluationRunRepository(Path(tmp))
            baseline = {
                "overall_metrics": {
                    "overall_score": 0.86,
                    "total_tests": 5,
                    "pass_rate": 0.8,
                    "regression_count": 0,
                    "avg_latency_ms": 800,
                }
            }
            repository.create_run("agent", {}, baseline)
            current = {
                "overall_metrics": {
                    "overall_score": 0.72,
                    "total_tests": 5,
                    "pass_rate": 0.6,
                    "regression_count": 0,
                    "avg_latency_ms": 1200,
                }
            }
            run = repository.create_run("agent", {}, current)
            self.assertEqual(run["comparison"]["status"], "regression")
            self.assertEqual(run["release_gate"]["status"], "review")
            self.assertTrue(run["release_gate"]["blockers"])


class EvolutionHarnessTests(unittest.TestCase):
    def test_generates_jd_coverage_and_self_evolution_proposals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = SkillRegistry()
            registry.log_file = Path(tmp) / "runs.jsonl"
            runtime = AgentRuntime(registry, SafetyGate())
            runtime.runtime_dir = Path(tmp) / "runtime"
            runtime.runtime_dir.mkdir()
            memory = ConversationMemory(Path(tmp) / "memory")
            repository = EvaluationRunRepository(Path(tmp) / "evals")
            repository.create_run(
                "agent",
                {},
                {"overall_metrics": {"overall_score": 0.9, "total_tests": 3, "pass_rate": 1, "regression_count": 0}},
            )

            report = EvolutionHarness(repository, runtime, registry, memory).report()
            self.assertGreaterEqual(report["maturity_score"], 70)
            self.assertEqual(report["jd_coverage"]["covered"], report["jd_coverage"]["total"])
            self.assertTrue(report["evolution_proposals"])
            self.assertTrue(report["harness_loops"])


if __name__ == "__main__":
    unittest.main()

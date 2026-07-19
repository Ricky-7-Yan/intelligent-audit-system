from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from services.agent_runtime import AgentRuntime
from services.agent_quality import AgentQualityDiagnostics
from services.audit_repository import AuditRunRepository
from services.conversation_memory import ConversationMemory
from services.evaluation_repository import EvaluationRunRepository
from services.evidence_analyzer import EvidenceAnalyzer
from services.evolution_harness import EvolutionHarness
from services.harness_control import HarnessControlPlane
from services.intent_router import HybridIntentRouter
from services.safety_gate import SafetyGate
from services.skill_registry import Skill, SkillRegistry
from web.main import collect_search_results


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
            self.assertLessEqual(context["context_budget"]["estimated_tokens"], context["context_budget"]["limit_tokens"])
            self.assertTrue(memory.delete_session("audit-session"))
            self.assertIsNone(memory.get_session("audit-session"))


class HarnessControlPlaneTests(unittest.TestCase):
    def test_requires_locked_surfaces_two_split_gate_and_human_review(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "project"
            for relative in ("services/evaluation_repository.py", "training/training_pipeline.py", "tests/test_agent_platform.py"):
                path = project / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"locked:{relative}", encoding="utf-8")
            control = HarnessControlPlane(Path(tmp) / "harness", project)
            candidate = control.create_candidate(
                {"proposal_id": "HNS-T", "title": "测试候选", "action": "优化工具契约", "validation": "双集无回归"},
                "services/skill_registry.py",
            )
            evaluated = control.evaluate_candidate(
                candidate["candidate_id"],
                {"quality": 0.80, "latency": 0.70},
                {"quality": 0.84, "latency": 0.70},
                {"quality": 0.82, "latency": 0.72},
                [{"name": "unit", "status": "pass"}],
            )
            self.assertEqual(evaluated["status"], "awaiting_human_review")
            approved = control.review_candidate(candidate["candidate_id"], "approve", "tester", "verified")
            self.assertEqual(approved["status"], "approved")
            self.assertGreaterEqual(control.summary()["event_count"], 3)
            self.assertTrue(control.archive_candidate(candidate["candidate_id"]))


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
            metrics = registry.metrics()
            self.assertGreaterEqual(metrics["skills"], 1)
            self.assertIn("open_circuits", metrics)
            self.assertIn("circuits", metrics)
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
            self.assertTrue(repository.delete_run(run["run_id"]))
            self.assertIsNone(repository.get_run(run["run_id"]))


class AuditEvidenceRepositoryTests(unittest.TestCase):
    def test_deletes_audit_runs_and_evidence_analyses(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_repository = AuditRunRepository(Path(tmp) / "audit")
            audit_run = audit_repository.create_run(
                {"audit_item": "ERP", "audit_type": "权限审计", "standard_type": "ISO27001"},
                {
                    "response": "审计结论",
                    "quality_gate": {"confidence": 0.82},
                    "risk_assessment": {"risk_level": "高", "risk_score": 0.8},
                    "compliance_check": {"compliance_score": 0.76},
                    "recommendations": [],
                    "control_matrix": [],
                    "audit_program": [],
                },
            )
            self.assertIsNotNone(audit_repository.get_run(audit_run["run_id"]))
            self.assertTrue(audit_repository.delete_run(audit_run["run_id"]))
            self.assertIsNone(audit_repository.get_run(audit_run["run_id"]))

            analyzer = EvidenceAnalyzer(Path(tmp) / "evidence")
            analysis = analyzer.analyze_file(
                "access.csv",
                b"user,role,status\nadmin,administrator,active\n",
                {"audit_item": "ERP", "audit_type": "权限审计"},
            )
            self.assertIsNotNone(analyzer.get_analysis(analysis["analysis_id"]))
            self.assertTrue(analyzer.delete_analysis(analysis["analysis_id"]))
            self.assertIsNone(analyzer.get_analysis(analysis["analysis_id"]))


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


class AgentQualityDiagnosticsTests(unittest.TestCase):
    def test_builds_interview_driven_quality_report_from_runtime_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = SkillRegistry()
            registry.log_file = Path(tmp) / "runs.jsonl"
            runtime = AgentRuntime(registry, SafetyGate())
            runtime.runtime_dir = Path(tmp) / "runtime"
            runtime.runtime_dir.mkdir()
            memory = ConversationMemory(Path(tmp) / "memory", compress_at=3, retain_recent=1)
            repository = EvaluationRunRepository(Path(tmp) / "evals")

            memory.record_turn(
                "interview-session",
                "Explain ERP permission audit with ISO27001 evidence.",
                "Use RAG evidence, tool traces, and quality gate.",
                {"intent": "control_testing", "agents": ["control_agent"]},
            )
            task = runtime.create_task(
                "Build an ERP audit plan with tool use and reflection",
                {"audit_item": "ERP permission"},
            )
            self.assertTrue(task["steps"])
            repository.create_run(
                "rag",
                {"cases": [{"question": "ERP access review"}]},
                {
                    "overall_score": 0.82,
                    "total_cases": 1,
                    "results": [{"overall": 0.82, "failure_modes": []}],
                },
            )
            registry.execute("audit.control_mapper", {"audit_item": "ERP", "standard": "ISO27001"})

            harness = HarnessControlPlane(Path(tmp) / "harness", Path(tmp))
            report = AgentQualityDiagnostics(repository, runtime, registry, memory, harness).report(
                {"total_documents": 4, "total_chunks": 12}
            )

            self.assertIn("overall_score", report)
            self.assertGreaterEqual(report["overall_score"], 50)
            self.assertEqual(
                {item["dimension_id"] for item in report["dimensions"]},
                {
                    "agent_runtime",
                    "rag_grounding",
                    "tool_mcp",
                    "evaluation_harness",
                    "memory_context",
                    "production_engineering",
                },
            )
            self.assertTrue(report["interview_pitch"])
            self.assertIn("top_tools", report["tool_use_diagnostics"])
            self.assertTrue(report["production_readiness"])
            harness_dimension = next(
                item for item in report["dimensions"] if item["dimension_id"] == "evaluation_harness"
            )
            self.assertTrue(any("人工审批" in evidence for evidence in harness_dimension["evidence"]))


class GlobalSearchTests(unittest.TestCase):
    def test_returns_static_commands_for_navigation(self) -> None:
        results = collect_search_results("评测", limit=5)
        self.assertTrue(results)
        self.assertTrue(any(item["href"] == "/training" for item in results))
        self.assertTrue(all({"type", "title", "subtitle", "href"}.issubset(item) for item in results))


if __name__ == "__main__":
    unittest.main()

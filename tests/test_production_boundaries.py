from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from config import SECURITY_CONFIG, WEB_CONFIG, runtime_configuration_issues
from rag.agentic_rag import HybridRetriever
from services.agent_runtime import AgentRuntime
from services.audit_repository import AuditRunRepository
from services.record_store import (
    RecordConflictError,
    SQLiteRecordStore,
    TenantBoundaryError,
)
from services.safety_gate import SafetyGate
from services.security import (
    AuditEventStore,
    Principal,
    bind_request_context,
    reset_request_context,
)
from services.skill_registry import Skill, SkillRegistry


def _principal(subject: str, tenant: str, projects: tuple[str, ...] = ()) -> Principal:
    return Principal(
        subject=subject,
        tenant_id=tenant,
        roles=("admin",),
        permissions=frozenset({"*"}),
        auth_method="bearer",
        project_ids=projects,
    )


class ProductionBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_record_store_rejects_cross_tenant_claim_and_stale_update(self) -> None:
        tokens = bind_request_context(_principal("alice", "tenant-a"), "REQ-A")
        try:
            store = SQLiteRecordStore(self.root / "records.sqlite3")
            with self.assertRaises(TenantBoundaryError):
                store.put("case", "1", {"tenant_id": "tenant-b", "value": 1})

            version = store.put("case", "1", {"tenant_id": "tenant-a", "value": 1})
            self.assertEqual(version, 1)
            version = store.put(
                "case",
                "1",
                {"tenant_id": "tenant-a", "value": 2},
                expected_version=1,
            )
            self.assertEqual(version, 2)
            with self.assertRaises(RecordConflictError):
                store.put(
                    "case",
                    "1",
                    {"tenant_id": "tenant-a", "value": 3},
                    expected_version=1,
                )
        finally:
            reset_request_context(tokens)

    def test_rag_project_filter_is_fail_closed(self) -> None:
        tokens = bind_request_context(_principal("alice", "tenant-a", ("project-a",)), "REQ-RAG")
        try:
            retriever = object.__new__(HybridRetriever)
            self.assertFalse(
                retriever._metadata_allowed(
                    {
                        "tenant_id": "tenant-a",
                        "project_id": "project-b",
                        "visibility": "project",
                    },
                    {},
                )
            )
            self.assertFalse(
                retriever._metadata_allowed(
                    {"tenant_id": "tenant-a", "visibility": "project"},
                    {"project_id": "project-a"},
                )
            )
            self.assertTrue(
                retriever._metadata_allowed(
                    {"tenant_id": "tenant-a", "visibility": "tenant_shared"},
                    {"project_id": "project-a"},
                )
            )
        finally:
            reset_request_context(tokens)

    def test_local_admin_mode_cannot_bind_public_interface(self) -> None:
        original_mode = SECURITY_CONFIG["mode"]
        original_host = WEB_CONFIG["host"]
        try:
            SECURITY_CONFIG["mode"] = "local"
            WEB_CONFIG["host"] = "0.0.0.0"
            self.assertTrue(
                any("SECURITY_MODE=local" in issue for issue in runtime_configuration_issues())
            )
        finally:
            SECURITY_CONFIG["mode"] = original_mode
            WEB_CONFIG["host"] = original_host

    def test_agent_contract_is_executable_not_descriptive(self) -> None:
        registry = SkillRegistry()
        runtime = AgentRuntime(registry, SafetyGate())
        plan_step = {
            "depends_on": [],
            "output_contract": ["scope", "objectives", "deliverables"],
            "termination_condition": "all fields are present",
        }
        step_record = {
            "status": "success",
            "output": {"scope": "ERP", "deliverables": ["report"]},
            "run_id": "SK-1",
            "attempts": 1,
            "safety_gate": {"status": "pass"},
        }
        evaluation = runtime._evaluate_step(
            plan_step,
            step_record,
            {"budgets": {"max_retries_per_step": 1}},
            set(),
        )
        assertions = {item["key"]: item["passed"] for item in evaluation["assertions"]}
        self.assertFalse(assertions["output_contract"])
        self.assertNotEqual(evaluation["status"], "pass")

    def test_direct_tool_call_rejects_invalid_output_schema(self) -> None:
        registry = SkillRegistry()
        registry._register(
            Skill(
                name="test.invalid_output",
                title="invalid",
                description="contract test",
                input_schema={"type": "object"},
                output_schema={
                    "type": "object",
                    "properties": {"required_result": {"type": "string"}},
                    "required": ["required_result"],
                },
                permissions=[],
                handler=lambda payload: {"display_only": True},
            )
        )
        run = registry.execute("test.invalid_output", {})
        self.assertEqual(run["status"], "failed")
        self.assertEqual(run["error_type"], "OutputValidationError")

    def test_signed_checkpoint_detects_rebuilt_hash_chain(self) -> None:
        tokens = bind_request_context(_principal("alice", "tenant-a"), "REQ-SIGN")
        try:
            store = AuditEventStore(self.root / "events.jsonl", signing_key="test-signing-key")
            store.append("create", "/api/audit", 200, {"run_id": "A"})
            self.assertTrue(store.verify()["valid"])

            event = json.loads(store.path.read_text(encoding="utf-8").strip())
            event["details"] = {"run_id": "ATTACKER-REBUILT"}
            event_without_hash = {key: value for key, value in event.items() if key != "event_hash"}
            event["event_hash"] = store._digest(event_without_hash)
            store.path.write_text(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")

            result = store.verify()
            self.assertFalse(result["valid"])
            self.assertEqual(result["checkpoint"], "stale")
        finally:
            reset_request_context(tokens)

    def test_project_membership_blocks_same_tenant_bystander(self) -> None:
        creator_tokens = bind_request_context(_principal("creator", "tenant-a"), "REQ-CREATE")
        try:
            repository = AuditRunRepository(self.root / "runs")
            run = repository.create_run(
                {"audit_item": "ERP", "audit_type": "access"},
                {"quality_gate": {}, "recommendations": [], "control_matrix": []},
            )
            run_id = run["run_id"]
            self.assertIsNotNone(repository.get_run(run_id))
        finally:
            reset_request_context(creator_tokens)

        bystander_tokens = bind_request_context(_principal("bystander", "tenant-a"), "REQ-READ")
        try:
            self.assertIsNone(repository.get_run(run_id))
        finally:
            reset_request_context(bystander_tokens)

    def test_delivery_verifier_checks_full_artifact_integrity(self) -> None:
        registry = SkillRegistry()

        def artifact(kind: str, content: dict[str, object], index: int) -> dict[str, object]:
            canonical = json.dumps(
                content,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            return {
                "artifact_id": f"ART-{index}",
                "type": kind,
                "source_run_id": f"SK-{index}",
                "producer": {"step_id": f"S-{index}", "skill": "test"},
                "content": content,
                "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            }

        artifacts = [
            artifact("audit", {"control_matrix": [{"control_id": "AC-1"}]}, 1),
            artifact("evidence", {"evidence_requests": [{"evidence": "log"}]}, 2),
            artifact("risk", {"severity": "high"}, 3),
            artifact(
                "risk",
                {
                    "owner": "control-owner",
                    "due_date": "2026-08-10",
                    "acceptance_criteria": "retest passes",
                },
                4,
            ),
        ]
        payload = {
            "audit_item": "ERP",
            "upstream_artifacts": artifacts,
            "upstream_steps": [{"step_id": "S-1", "status": "success"}],
        }
        self.assertEqual(registry._delivery_verifier(payload)["verdict"], "pass")
        artifacts[0]["content"] = {"control_matrix": []}
        result = registry._delivery_verifier(payload)
        self.assertEqual(result["verdict"], "human_review")
        self.assertFalse(
            next(item for item in result["checks"] if item["check"] == "artifact_integrity")[
                "passed"
            ]
        )

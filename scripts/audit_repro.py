"""Audit repro probes for AuditPilot findings.

Run from the project root with the venv interpreter:

    & "$env:TEMP\auditpilot_venv\Scripts\python.exe" scripts/audit_repro.py

Each probe asserts a contract that the README / docs promise and verifies
that the current implementation honors it. When the contract is broken the
probe prints FAIL and the script exits with a non-zero status.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agents.audit_agent import AuditAgent, RISK_KEYWORDS  # noqa: E402
from services.agent_runtime import AgentRuntime  # noqa: E402
from services.evaluation_repository import EvaluationRunRepository  # noqa: E402
from services.skill_registry import SkillRegistry  # noqa: E402
from services.safety_gate import SafetyGate  # noqa: E402


SEPARATOR = "=" * 72
_RESULTS: list[tuple[str, bool, str]] = []


def banner(title: str) -> None:
    print(SEPARATOR)
    print(title)
    print(SEPARATOR)


def expect(condition: bool, message: str, detail: str = "") -> None:
    status = "OK" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if detail:
        print(f"         {detail}")
    _RESULTS.append((message, condition, detail))


def probe_p0_1_dead_code() -> None:
    """AuditAgent._compose_response must surface the executive summary
    AND the structured risk/remediation tables."""
    banner("P0-1  compose_response surfaces the full summary + tables")
    agent = AuditAgent(enable_llm=False, enable_external_tools=False)
    result = agent.process_audit_query("请对 ERP 系统进行权限审计", prefer_llm=False)
    response = result.get("response", "")
    expect("审计对象" in response, "审计对象 字段应出现在摘要",
           f"response starts with: {response[:80]!r}")
    expect("优先动作" in response, "应包含 ‘优先动作’ 字段",
           "missing '优先动作' segment; reported bug: dead code after return")
    expect("高风险控制缺陷" in response, "应包含高风险控制缺陷表",
           "tables dropped from deterministic response")
    expect("整改动作计划" in response, "应包含整改动作计划表",
           "tables dropped from deterministic response")


def probe_p0_6_redact_overreach() -> None:
    """SkillRegistry._redact should not treat 'sk-001' as a credential,
    but should still redact real OpenAI keys and Bearer tokens."""
    banner("P0-6  redact is precise about credentials")
    with tempfile.TemporaryDirectory() as tmp:
        registry = SkillRegistry()
        registry.log_file = Path(tmp) / "runs.jsonl"
        redacted = registry._redact(
            {
                "criteria": "设备编号 sk-001",
                "note": "Bearer token should still be redacted",
                "real_key": "sk-" + ("A" * 24),
                "sk_artifact": "sk-001",
            }
        )
        expect(redacted["criteria"] == "设备编号 sk-001",
               "非凭据字符串 sk-001 不应被 [REDACTED]",
               f"actual: {redacted['criteria']!r}")
        expect(redacted["real_key"] == "[REDACTED]",
               "真实凭据仍应被 [REDACTED]",
               f"actual: {redacted['real_key']!r}")
        expect(redacted["note"] == "Bearer token should still be redacted",
               "Bearer 字样不应导致整段被脱敏",
               f"actual: {redacted['note']!r}")
        expect(redacted["sk_artifact"] == "sk-001",
               "短前缀 'sk-001' 是业务编号不是凭据",
               f"actual: {redacted['sk_artifact']!r}")


def probe_p0_8_baseline_meaning() -> None:
    """EvaluationRunRepository should support a locked baseline distinct
    from 'previous run'."""
    banner("P0-8  baseline supports a locked reference run")
    with tempfile.TemporaryDirectory() as tmp:
        repo = EvaluationRunRepository(Path(tmp) / "evals")
        first = repo.create_run(
            "task_component",
            {},
            {
                "summary": {
                    "overall_score": 0.90,
                    "total_tests": 6,
                    "pass_rate": 0.95,
                    "critical_failures": [],
                }
            },
        )
        second = repo.create_run(
            "task_component",
            {},
            {
                "summary": {
                    "overall_score": 0.78,
                    "total_tests": 6,
                    "pass_rate": 0.78,
                    "critical_failures": [],
                }
            },
        )
        expect(
            (second.get("comparison") or {}).get("baseline_run_id") == first["run_id"],
            "首次 baseline_created 时不能回环到自身",
            f"baseline_run_id: {(second.get('comparison') or {}).get('baseline_run_id')}",
        )
        repo.mark_as_baseline(first["run_id"])
        third = repo.create_run(
            "task_component",
            {},
            {
                "summary": {
                    "overall_score": 0.86,
                    "total_tests": 6,
                    "pass_rate": 0.90,
                    "critical_failures": [],
                }
            },
        )
        expect(
            (third.get("comparison") or {}).get("baseline_locked") is True,
            "锁定基线应被识别为 locked",
            f"baseline_locked: {(third.get('comparison') or {}).get('baseline_locked')}",
        )
        expect(
            (third.get("comparison") or {}).get("baseline_run_id") == first["run_id"],
            "锁定基线应覆盖 'previous run' 决策",
            f"baseline_run_id: {(third.get('comparison') or {}).get('baseline_run_id')}",
        )


def probe_p0_9_integrity_digest_self_reference() -> None:
    """episode_package integrity digest should be a stable hash of the
    canonical payload without the integrity field itself."""
    banner("P0-9  integrity digest is self-consistent")
    with tempfile.TemporaryDirectory() as tmp:
        registry = SkillRegistry()
        registry.log_file = Path(tmp) / "runs.jsonl"
        runtime = AgentRuntime(registry, SafetyGate())
        runtime.runtime_dir = Path(tmp) / "runtime"
        runtime.runtime_dir.mkdir()
        task = runtime.create_task("生成 ERP 权限审计计划", {"audit_item": "ERP 权限"})
        episode = runtime.episode_package(task["task_id"])
        integrity = episode.get("integrity", {})
        digest = integrity.get("digest")
        algorithm = integrity.get("algorithm")
        cleaned = {key: value for key, value in episode.items() if key != "integrity"}
        canonical = json.dumps(cleaned, ensure_ascii=False, sort_keys=True, default=str)
        expected_digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        expect(
            isinstance(digest, str) and len(digest) == 64,
            "digest is a sha256 hex string",
            f"algorithm={algorithm!r}, len={len(digest) if isinstance(digest, str) else 'n/a'}",
        )
        expect(
            digest == expected_digest,
            "digest 等于基于去掉 integrity 字段的稳定散列",
            f"plain digest: {digest!r}; expected: {expected_digest!r}",
        )


def probe_p1_3_risk_columns_meaningful() -> None:
    """Deterministic response renders 'impact/likelihood/control_gap'
    with non-placeholder content."""
    banner("P1-3  risk table columns are populated")
    agent = AuditAgent(enable_llm=False, enable_external_tools=False)
    result = agent.process_audit_query("请对 ERP 权限管理进行审计", prefer_llm=False)
    response = result.get("response", "")
    expect(
        RISK_KEYWORDS and all("score" in entry and "risk" in entry for entry in RISK_KEYWORDS.values()),
        "RISK_KEYWORDS schema 已包含 score/risk/domain",
        f"keys={list(next(iter(RISK_KEYWORDS.values())).keys())}",
    )
    expect(
        "影响" in response and "可能性" in response and "控制缺口" in response,
        "风险表格应包含影响/可能性/控制缺口三维",
        "schema/模板尚未填充这些字段",
    )
    risk_assessment = result.get("risk_assessment", {})
    identified = risk_assessment.get("identified_risks", [])
    expect(
        bool(identified) and all(
            isinstance(item.get("impact"), str) and "-" not in item.get("impact", "")
            for item in identified
        ),
        "identified_risks 每个条目都应包含 impact 文案",
        f"first item: {identified[0] if identified else 'n/a'}",
    )


def probe_p3_3_database_seed_duplication() -> None:
    """database/init_db.py should be idempotent across reruns."""
    banner("P3-3  database/init_db idempotent seeds")
    init_path = PROJECT_ROOT / "database" / "init_db.py"
    text = init_path.read_text(encoding="utf-8")
    executemany_block = "INSERT INTO audit_standards"
    expect(
        "INSERT IGNORE" in text,
        "INSERT IGNORE 已加入种子插入",
        "重复执行 init_db 会持续向 audit_standards 插入 5 条种子",
    )
    expect(
        "UNIQUE KEY" in text and "uniq_standard_name" in text,
        "audit_standards 已设置 uniq_standard_name 唯一键",
        "缺少唯一键会导致 INSERT IGNORE 在历史脏数据上失效",
    )
    expect(
        executemany_block not in text or "INSERT IGNORE INTO audit_standards" in text,
        "审计标准 stats 写入不再是 INSERT ... VALUES",
        "executor must use INSERT IGNORE for audit_standards",
    )


def main() -> int:
    os.environ.setdefault("AUDIT_DISABLE_LLM", "1")
    os.environ.setdefault("RAG_DISABLE_LLM", "1")
    os.environ.setdefault("RAG_DISABLE_EMBEDDINGS", "1")
    os.environ.setdefault("RAG_LIGHT_MODE", "0")

    probe_p0_1_dead_code()
    probe_p0_6_redact_overreach()
    probe_p0_8_baseline_meaning()
    probe_p0_9_integrity_digest_self_reference()
    probe_p1_3_risk_columns_meaningful()
    probe_p3_3_database_seed_duplication()

    print(SEPARATOR)
    total = len(_RESULTS)
    failed = sum(1 for _, ok, _ in _RESULTS if not ok)
    print(f"summary: {total - failed}/{total} assertions passed")
    if failed:
        print("failed probes:")
        for message, ok, detail in _RESULTS:
            if not ok:
                print(f"  - {message}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

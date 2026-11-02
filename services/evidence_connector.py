"""Governed read-only evidence snapshot connectors."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from config import PATHS
from services.evidence_analyzer import EvidenceAnalyzer
from services.record_store import SQLiteRecordStore
from services.security import current_tenant_id


CONNECTOR_PROFILES: Dict[str, Dict[str, Any]] = {
    "iam_access_snapshot": {
        "name": "IAM / ERP 权限快照",
        "description": "只读导入账号、角色、状态和授权关系，用于越权、离职账号与职责分离检查。",
        "accepted_extensions": [".csv", ".tsv", ".json"],
        "required_field_groups": [["user", "username", "账号", "用户"], ["role", "权限", "角色"]],
        "source_examples": ["IAM 导出", "ERP 用户角色报表", "AD 账号快照"],
    },
    "change_ticket_snapshot": {
        "name": "变更工单快照",
        "description": "只读导入需求、审批、测试、发布和回退字段，用于识别未经审批或证据断裂的变更。",
        "accepted_extensions": [".csv", ".tsv", ".json"],
        "required_field_groups": [["change", "ticket", "变更", "工单"], ["approval", "审批", "approver"]],
        "source_examples": ["ITSM 变更导出", "发布平台工单", "紧急变更清单"],
    },
    "generic_audit_snapshot": {
        "name": "通用审计日志快照",
        "description": "只读导入结构化业务或安全日志，保留来源、哈希和解析结果。",
        "accepted_extensions": [".csv", ".tsv", ".json", ".log", ".txt"],
        "required_field_groups": [],
        "source_examples": ["数据库审计日志", "应用操作日志", "业务台账"],
    },
}


class EvidenceConnectorService:
    def __init__(self, analyzer: EvidenceAnalyzer, root: Path | None = None) -> None:
        self.analyzer = analyzer
        self.root = root or (PATHS["data"] / "connector_runs")
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteRecordStore(self.root / ".records.sqlite3")

    def catalog(self) -> List[Dict[str, Any]]:
        return [
            {
                "connector_id": connector_id,
                **profile,
                "mode": "read_only_snapshot",
                "idempotency": "sha256_per_tenant_and_source",
            }
            for connector_id, profile in CONNECTOR_PROFILES.items()
        ]

    def ingest(
        self,
        *,
        connector_id: str,
        source_system: str,
        file_name: str,
        content: bytes,
        sha256: str,
        audit_context: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        profile = CONNECTOR_PROFILES.get(connector_id)
        if not profile:
            raise ValueError("不支持的连接器类型")
        extension = Path(file_name).suffix.lower()
        if extension not in profile["accepted_extensions"]:
            raise ValueError(f"{profile['name']} 不支持 {extension or '无扩展名'} 文件")
        source_system = " ".join(str(source_system or "").split())[:160]
        if not source_system:
            raise ValueError("source_system 不能为空")

        idempotency_key = f"{connector_id}:{source_system.lower()}:{sha256.lower()}"
        run_id = f"CR-{hashlib.sha256(idempotency_key.encode('utf-8')).hexdigest()[:20].upper()}"
        duplicate = self.store.get("connector_run", run_id)
        if duplicate and self.analyzer.get_analysis(str(duplicate.get("analysis_id") or "")):
            return {**duplicate, "duplicate": True}
        if duplicate:
            self.store.delete("connector_run", run_id)

        context = {
            **(audit_context or {}),
            "connector_id": connector_id,
            "connector_mode": "read_only_snapshot",
            "source_system": source_system,
            "source_sha256": sha256,
        }
        analysis = self.analyzer.analyze_file(file_name, content, context)
        fields = [str(field).lower() for field in analysis.get("profile", {}).get("fields", [])]
        field_checks = []
        for alternatives in profile.get("required_field_groups", []):
            matched = next((candidate for candidate in alternatives if any(candidate.lower() in field for field in fields)), None)
            field_checks.append({"alternatives": alternatives, "matched": matched, "passed": bool(matched)})
        schema_ready = all(item["passed"] for item in field_checks)
        record = {
            "connector_run_id": run_id,
            "tenant_id": current_tenant_id(),
            "connector_id": connector_id,
            "connector_name": profile["name"],
            "mode": "read_only_snapshot",
            "source_system": source_system,
            "source_file": file_name,
            "source_sha256": sha256,
            "idempotency_key": idempotency_key,
            "created_at": datetime.now().isoformat(),
            "status": "ready" if schema_ready else "review",
            "schema_validation": {"passed": schema_ready, "checks": field_checks},
            "analysis_id": analysis["analysis_id"],
            "row_count": analysis.get("profile", {}).get("rows", 0),
            "field_count": analysis.get("profile", {}).get("field_count", 0),
            "risk_count": len(analysis.get("risk_signals", [])),
            "control_count": len(analysis.get("mapped_controls", [])),
            "quality_gate": analysis.get("quality_gate", {}),
            "audit_context": audit_context or {},
            "duplicate": False,
        }
        version = self.store.put("connector_run", run_id, record)
        record["_storage_version"] = version
        return record

    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        return self.store.list("connector_run", limit=max(1, min(int(limit), 200)))

    def get_run(self, connector_run_id: str) -> Dict[str, Any] | None:
        return self.store.get("connector_run", connector_run_id)

    def delete_run(self, connector_run_id: str) -> bool:
        """Delete connector metadata without mutating the upstream source snapshot."""
        return self.store.delete("connector_run", connector_run_id)

    def health(self) -> Dict[str, Any]:
        health = self.store.health()
        return {**health, "connector_profiles": len(CONNECTOR_PROFILES)}

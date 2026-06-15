"""Audit-industry delivery package generation."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from services.audit_repository import AuditRunRepository


class AuditDeliveryService:
    def __init__(self, repository: AuditRunRepository) -> None:
        self.repository = repository

    def build_package(self, run_id: str) -> Optional[Dict[str, Any]]:
        record = self.repository.get_run(run_id)
        if not record:
            return None
        result = record.get("result", {})
        request = record.get("request", {})
        controls = result.get("control_matrix", [])
        evidence = result.get("evidence_pack", [])
        procedures = result.get("audit_program", [])
        findings = result.get("findings", [])
        tasks = record.get("remediation_tasks", [])

        return {
            "run_id": run_id,
            "generated_at": datetime.now().isoformat(),
            "engagement": {
                "audit_item": request.get("audit_item"),
                "audit_type": request.get("audit_type"),
                "standard": request.get("standard_type"),
                "risk_level": request.get("risk_level"),
                "status": record.get("status"),
            },
            "workpaper_index": self._workpaper_index(result),
            "evidence_request_list": self._evidence_request_list(result),
            "control_test_plan": self._control_test_plan(controls, procedures),
            "finding_tracker": self._finding_tracker(findings, tasks),
            "quality_review": result.get("quality_gate", {}),
            "signoff": {
                "prepared_by": "智能审计 Agent",
                "reviewer": "审计经理",
                "review_required": bool(result.get("quality_gate", {}).get("escalation_required")),
                "reviews": record.get("reviews", []),
            },
        }

    def _workpaper_index(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows = [
            {"ref": "WP-00", "name": "审计范围与目标", "source": "task_plan", "owner": "审计经理"},
            {"ref": "WP-10", "name": "RAG 证据检索记录", "source": "evidence_pack", "owner": "审计员"},
            {"ref": "WP-20", "name": "控制矩阵", "source": "control_matrix", "owner": "控制测试员"},
            {"ref": "WP-30", "name": "审计程序与抽样计划", "source": "audit_program", "owner": "审计员"},
            {"ref": "WP-40", "name": "审计发现与整改计划", "source": "findings", "owner": "审计经理"},
            {"ref": "WP-50", "name": "质量门与复核记录", "source": "quality_gate", "owner": "复核人"},
        ]
        for index, control in enumerate(result.get("control_matrix", []), start=1):
            rows.append(
                {
                    "ref": f"WP-20-{index:02d}",
                    "name": f"{control.get('control_id')} {control.get('domain')} 控制测试",
                    "source": control.get("control_id"),
                    "owner": "控制测试员",
                }
            )
        return rows

    def _evidence_request_list(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        requests = []
        for index, item in enumerate(result.get("evidence_pack", []), start=1):
            requests.append(
                {
                    "id": f"EV-{index:02d}",
                    "source": item.get("source"),
                    "summary": item.get("summary"),
                    "usage": item.get("usage"),
                    "status": "已获取" if item.get("type") != "heuristic" else "待补充",
                }
            )
        missing = result.get("quality_gate", {}).get("missing_evidence", [])
        for index, item in enumerate(missing, start=len(requests) + 1):
            requests.append({"id": f"EV-{index:02d}", "source": "现场取证", "summary": item, "usage": "补齐质量门缺口", "status": "待收集"})
        return requests

    def _control_test_plan(self, controls: List[Dict[str, Any]], procedures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        procedure_by_control = {item.get("control_id"): item for item in procedures}
        rows = []
        for control in controls:
            procedure = procedure_by_control.get(control.get("control_id"), {})
            rows.append(
                {
                    "control_id": control.get("control_id"),
                    "domain": control.get("domain"),
                    "test_procedure": control.get("test_procedure"),
                    "assertion": procedure.get("assertion"),
                    "sample_method": procedure.get("method"),
                    "evidence_required": control.get("evidence_required", []),
                    "workpaper_ref": procedure.get("workpaper_ref"),
                }
            )
        return rows

    def _finding_tracker(self, findings: List[Dict[str, Any]], tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        rows = []
        for finding in findings:
            related_tasks = [task for task in tasks if finding.get("finding_id", "") in task.get("description", "")] or tasks[:2]
            rows.append(
                {
                    "finding_id": finding.get("finding_id"),
                    "title": finding.get("title"),
                    "severity": finding.get("severity"),
                    "condition": finding.get("condition"),
                    "recommendation": finding.get("recommendation"),
                    "tasks": [{"task_id": task.get("task_id"), "status": task.get("status"), "owner": task.get("owner")} for task in related_tasks],
                }
            )
        return rows

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
                "lifecycle_stage": record.get("lifecycle_stage"),
                "business_context": request.get("business_context"),
                "audit_scope": request.get("audit_scope"),
                "audit_period": request.get("audit_period"),
                "key_questions": request.get("key_questions"),
                "existing_evidence": request.get("existing_evidence"),
            },
            "workpaper_index": self._workpaper_index(result),
            "evidence_request_list": self._evidence_request_list(record, result),
            "control_test_plan": self._control_test_plan(record, controls, procedures),
            "finding_tracker": self._finding_tracker(findings, tasks),
            "interview_plan": self._interview_plan(result),
            "fieldwork_calendar": self._fieldwork_calendar(result),
            "quality_review": result.get("quality_gate", {}),
            "event_log": record.get("events", []),
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

    def _evidence_request_list(self, record: Dict[str, Any], result: Dict[str, Any]) -> List[Dict[str, Any]]:
        requests = []
        for item in record.get("evidence_requests", []):
            requests.append(
                {
                    "id": item.get("request_id"),
                    "source": item.get("source"),
                    "summary": item.get("evidence"),
                    "usage": item.get("usage"),
                    "owner": item.get("owner"),
                    "priority": item.get("priority"),
                    "status": item.get("status"),
                }
            )
        if requests:
            return requests
        for index, item in enumerate(result.get("evidence_pack", []), start=1):
            requests.append(
                {
                    "id": f"EV-{index:02d}",
                    "source": item.get("source"),
                    "summary": item.get("summary"),
                    "usage": item.get("usage"),
                    "owner": "审计员",
                    "priority": "中",
                    "status": "已获取" if item.get("type") != "heuristic" else "待补充",
                }
            )
        return requests

    def _control_test_plan(self, record: Dict[str, Any], controls: List[Dict[str, Any]], procedures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if record.get("control_tests"):
            return record["control_tests"]
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
                    "result": "待执行",
                    "exception_rule": "发现重大例外时扩大样本并升级复核。",
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

    def _interview_plan(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        domains = []
        for control in result.get("control_matrix", []):
            domain = control.get("domain")
            if domain and domain not in domains:
                domains.append(domain)
        return [
            {
                "topic": domain,
                "interviewee": "流程负责人 / 系统管理员 / 控制责任人",
                "questions": [
                    f"{domain} 控制的责任边界和审批链路是什么？",
                    "关键例外如何审批、记录和复核？",
                    "最近一次控制执行证据存放在哪里？",
                ],
            }
            for domain in domains[:6]
        ]

    def _fieldwork_calendar(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks = result.get("task_plan", [])
        calendar = []
        for index, task in enumerate(tasks, start=1):
            calendar.append(
                {
                    "day": f"D+{index}",
                    "activity": task.get("name"),
                    "owner": task.get("owner"),
                    "output": task.get("objective"),
                }
            )
        return calendar

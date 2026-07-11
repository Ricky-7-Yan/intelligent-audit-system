"""Persistent audit run storage and report rendering."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS


TASK_STATUSES = ["未开始", "进行中", "待验证", "已完成", "已关闭"]
RUN_LIFECYCLE = ["立项", "取证", "测试", "复核", "报告", "整改跟踪", "关闭"]


class AuditRunRepository:
    def __init__(self, root: Optional[Path] = None) -> None:
        self.root = root or (PATHS["data"] / "audit_runs")
        self.root.mkdir(parents=True, exist_ok=True)

    def create_run(self, request: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        run_id = f"AR-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        record = {
            "run_id": run_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "待复核" if result.get("quality_gate", {}).get("escalation_required") else "待现场验证",
            "lifecycle_stage": "取证",
            "request": request,
            "result": result,
            "remediation_tasks": self._build_remediation_tasks(run_id, result),
            "evidence_requests": self._build_evidence_requests(run_id, result),
            "control_tests": self._build_control_tests(result),
            "evidence_analyses": [],
            "reviews": [],
            "events": [{"at": datetime.now().isoformat(), "type": "created", "message": "审计项目已创建"}],
        }
        self._write(record)
        return record

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        records = []
        for record in self.iter_records(limit=1000):
            result = record.get("result", {})
            records.append(
                {
                    "run_id": record.get("run_id"),
                    "created_at": record.get("created_at"),
                    "updated_at": record.get("updated_at"),
                    "status": self._clean_legacy(record.get("status")),
                    "lifecycle_stage": self._clean_legacy(record.get("lifecycle_stage", "取证")),
                    "audit_item": self._display_text(record.get("request", {}).get("audit_item"), "历史审计档案"),
                    "audit_type": self._display_text(record.get("request", {}).get("audit_type"), "综合审计"),
                    "risk_level": self._clean_legacy(result.get("risk_assessment", {}).get("risk_level")),
                    "risk_score": result.get("risk_assessment", {}).get("risk_score"),
                    "quality_confidence": result.get("quality_gate", {}).get("confidence"),
                    "compliance_score": result.get("compliance_check", {}).get("compliance_score"),
                }
            )
        records.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return records[:limit]

    def iter_records(self, limit: int = 200) -> List[Dict[str, Any]]:
        records = []
        for path in self.root.glob("*.json"):
            try:
                records.append(self._normalize_record(json.loads(path.read_text(encoding="utf-8"))))
            except Exception:
                continue
        records.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return records[:limit]

    def task_summary(self) -> Dict[str, Any]:
        status_counts: Dict[str, int] = {}
        open_tasks = 0
        overdue_tasks = 0
        for record in self.iter_records(limit=1000):
            created_at = self._parse_date(record.get("created_at"))
            for task in record.get("remediation_tasks", []):
                status = self._clean_legacy(task.get("status", "未知"))
                status_counts[status] = status_counts.get(status, 0) + 1
                if status not in {"已完成", "已关闭", "done", "closed"}:
                    open_tasks += 1
                    due_days = int(task.get("due_days") or 0)
                    if created_at and due_days >= 0 and (datetime.now() - created_at).days > due_days:
                        overdue_tasks += 1
        return {"open_tasks": open_tasks, "overdue_tasks": overdue_tasks, "status_distribution": status_counts}

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(run_id)
        if not path.exists():
            return None
        return self._normalize_record(json.loads(path.read_text(encoding="utf-8")))

    def delete_run(self, run_id: str) -> bool:
        path = self._path(run_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def add_review(self, run_id: str, reviewer: str, decision: str, comment: str) -> Optional[Dict[str, Any]]:
        record = self.get_run(run_id)
        if not record:
            return None
        review = {
            "reviewer": reviewer or "复核人",
            "decision": decision,
            "comment": comment,
            "created_at": datetime.now().isoformat(),
        }
        record.setdefault("reviews", []).append(review)
        record["status"] = "已通过" if decision == "approve" else "需整改" if decision == "reject" else "待补充证据"
        record["lifecycle_stage"] = "报告" if decision == "approve" else "复核"
        self._append_event(record, "review", f"{review['reviewer']} 提交复核结论：{record['status']}")
        self._write(record)
        return record

    def update_task(self, run_id: str, task_id: str, status: str, owner: str = "", note: str = "") -> Optional[Dict[str, Any]]:
        record = self.get_run(run_id)
        if not record:
            return None
        for task in record.get("remediation_tasks", []):
            if task.get("task_id") == task_id:
                task["status"] = self._clean_legacy(status)
                if owner:
                    task["owner"] = owner
                if note:
                    task.setdefault("notes", []).append({"note": note, "at": datetime.now().isoformat()})
                task["updated_at"] = datetime.now().isoformat()
                record["lifecycle_stage"] = "整改跟踪"
                self._append_event(record, "task_update", f"整改任务 {task_id} 更新为 {task['status']}")
                self._write(record)
                return record
        return None

    def update_evidence_request(self, run_id: str, request_id: str, status: str, owner: str = "", note: str = "") -> Optional[Dict[str, Any]]:
        record = self.get_run(run_id)
        if not record:
            return None
        for item in record.get("evidence_requests", []):
            if item.get("request_id") == request_id:
                item["status"] = self._clean_legacy(status)
                if owner:
                    item["owner"] = owner
                if note:
                    item.setdefault("notes", []).append({"note": note, "at": datetime.now().isoformat()})
                item["updated_at"] = datetime.now().isoformat()
                record["lifecycle_stage"] = "取证"
                self._append_event(record, "evidence_update", f"证据请求 {request_id} 更新为 {item['status']}")
                self._write(record)
                return record
        return None

    def update_control_test(self, run_id: str, control_id: str, result: str, tester: str = "", exception: str = "") -> Optional[Dict[str, Any]]:
        record = self.get_run(run_id)
        if not record:
            return None
        for item in record.get("control_tests", []):
            if item.get("control_id") == control_id:
                item["result"] = result
                if tester:
                    item["tester"] = tester
                if exception:
                    item.setdefault("exceptions", []).append({"exception": exception, "at": datetime.now().isoformat()})
                item["updated_at"] = datetime.now().isoformat()
                record["lifecycle_stage"] = "测试"
                self._append_event(record, "control_test", f"控制 {control_id} 测试结果更新为 {result}")
                self._write(record)
                return record
        return None

    def attach_evidence_analysis(self, run_id: str, analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        record = self.get_run(run_id)
        if not record:
            return None
        analysis_id = analysis.get("analysis_id")
        if not analysis_id:
            return record

        attached = record.setdefault("evidence_analyses", [])
        if not any(item.get("analysis_id") == analysis_id for item in attached):
            attached.append(
                {
                    "analysis_id": analysis_id,
                    "file_name": analysis.get("file_name"),
                    "created_at": analysis.get("created_at"),
                    "risk_count": len(analysis.get("risk_signals", [])),
                    "control_count": len(analysis.get("mapped_controls", [])),
                    "quality_gate": analysis.get("quality_gate", {}),
                    "workpaper_ref": f"WP-EA-{len(attached) + 1:02d}",
                }
            )

        existing_evidence = {item.get("evidence") for item in record.get("evidence_requests", [])}
        for request in analysis.get("evidence_requests", []):
            evidence = request.get("evidence")
            if not evidence or evidence in existing_evidence:
                continue
            record.setdefault("evidence_requests", []).append(
                {
                    "request_id": f"{run_id}-EA-EV-{len(record.get('evidence_requests', [])) + 1:02d}",
                    "evidence": evidence,
                    "source": f"证据分析 {analysis_id}",
                    "usage": request.get("usage", "补充证据分析识别的底稿"),
                    "owner": "审计员",
                    "status": "待收集",
                    "priority": request.get("priority", "中"),
                    "required_fields": request.get("required_fields", []),
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "notes": [],
                }
            )
            existing_evidence.add(evidence)

        existing_controls = {item.get("control_id") for item in record.get("control_tests", [])}
        for control in analysis.get("mapped_controls", []):
            control_id = control.get("control_id")
            if not control_id or control_id in existing_controls:
                continue
            record.setdefault("control_tests", []).append(
                {
                    "control_id": control_id,
                    "domain": control.get("domain"),
                    "assertion": "完整性、授权、可追溯性",
                    "procedure": control.get("test_procedure"),
                    "workpaper_ref": f"WP-EA-{analysis_id}",
                    "sample_method": "基于证据文件分析结果执行定向抽样",
                    "result": "待执行",
                    "tester": "审计员",
                    "exceptions": [],
                    "updated_at": datetime.now().isoformat(),
                }
            )
            existing_controls.add(control_id)

        record["lifecycle_stage"] = "取证"
        record["status"] = "待现场验证"
        self._append_event(record, "evidence_analysis_attached", f"证据分析 {analysis_id} 已归档到审计项目")
        self._write(record)
        return record

    def render_markdown_report(self, run_id: str) -> Optional[str]:
        record = self.get_run(run_id)
        if not record:
            return None

        result = record.get("result", {})
        risk = result.get("risk_assessment", {})
        compliance = result.get("compliance_check", {})
        quality = result.get("quality_gate", {})
        request = record.get("request", {})

        lines = [
            f"# 审脉 AuditPilot 审计报告 - {run_id}",
            "",
            f"- 审计对象：{request.get('audit_item', '')}",
            f"- 审计类型：{request.get('audit_type', '')}",
            f"- 参考标准：{request.get('standard_type', '')}",
            f"- 生成时间：{record.get('created_at', '')}",
            f"- 当前状态：{record.get('status', '')}",
            f"- 项目阶段：{record.get('lifecycle_stage', '')}",
            "",
            "## 结论摘要",
            "",
            result.get("response", ""),
            "",
            "## 风险与合规",
            "",
            f"- 剩余风险等级：{risk.get('risk_level', '')}",
            f"- 风险评分：{risk.get('risk_score', '')}",
            f"- 固有风险评分：{risk.get('inherent_risk_score', '')}",
            f"- 控制抵减：{risk.get('control_reduction', '')}",
            f"- 合规评分：{compliance.get('compliance_score', '')}",
            f"- 控制成熟度均值：{compliance.get('control_maturity_avg', '')}",
            "",
            "## 质量门",
            "",
            f"- 状态：{quality.get('status', '')}",
            f"- 置信度：{quality.get('confidence', '')}",
            f"- 证据扎实度：{quality.get('groundedness', '')}",
            f"- 控制覆盖：{quality.get('control_coverage', '')}",
            f"- 缺失证据：{'、'.join(quality.get('missing_evidence', []) or [])}",
            "",
            "## 控制矩阵",
            "",
            "| 控制 | 领域 | 成熟度 | 状态 | 测试程序 |",
            "| --- | --- | --- | --- | --- |",
        ]

        for control in result.get("control_matrix", []):
            lines.append(
                f"| {control.get('control_id', '')} | {control.get('domain', '')} | "
                f"{control.get('maturity_level', '')} | {control.get('status', '')} | "
                f"{self._clean_table(control.get('test_procedure', ''))} |"
            )

        lines.extend(["", "## 审计程序", "", "| 步骤 | 控制 | 认定 | 方法 | 底稿索引 |", "| --- | --- | --- | --- | --- |"])
        for procedure in result.get("audit_program", []):
            lines.append(
                f"| {procedure.get('step_id', '')} | {procedure.get('control_id', '')} | "
                f"{procedure.get('assertion', '')} | {self._clean_table(procedure.get('method', ''))} | "
                f"{procedure.get('workpaper_ref', '')} |"
            )

        sampling = result.get("sampling_plan", {})
        if sampling:
            lines.extend(
                [
                    "",
                    "## 抽样计划",
                    "",
                    f"- 总体：{sampling.get('population', '')}",
                    f"- 期间：{sampling.get('period', '')}",
                    f"- 方法：{sampling.get('method', '')}",
                    f"- 样本量：{sampling.get('sample_size', '')}",
                    f"- 分层：{'、'.join(sampling.get('strata', []) or [])}",
                    f"- 例外处理：{sampling.get('exception_handling', '')}",
                ]
            )

        lines.extend(["", "## 审计发现草稿", ""])
        findings = result.get("findings", [])
        if not findings:
            lines.append("当前未形成重大审计发现草稿。")
        for finding in findings:
            lines.extend(
                [
                    f"### {finding.get('finding_id', '')} {finding.get('title', '')}",
                    "",
                    f"- 严重程度：{finding.get('severity', '')}",
                    f"- 现状：{finding.get('condition', '')}",
                    f"- 标准：{finding.get('criteria', '')}",
                    f"- 原因：{finding.get('cause', '')}",
                    f"- 影响：{finding.get('effect', '')}",
                    f"- 建议：{finding.get('recommendation', '')}",
                    "",
                ]
            )

        lines.extend(["", "## 证据请求中心", "", "| 请求 | 证据 | 责任人 | 状态 | 用途 |", "| --- | --- | --- | --- | --- |"])
        for item in record.get("evidence_requests", []):
            lines.append(f"| {item.get('request_id', '')} | {item.get('evidence', '')} | {item.get('owner', '')} | {item.get('status', '')} | {self._clean_table(item.get('usage', ''))} |")

        lines.extend(["", "## 控制测试工作台", "", "| 控制 | 领域 | 结果 | 测试人 | 底稿 |", "| --- | --- | --- | --- | --- |"])
        for item in record.get("control_tests", []):
            lines.append(f"| {item.get('control_id', '')} | {item.get('domain', '')} | {item.get('result', '')} | {item.get('tester', '')} | {item.get('workpaper_ref', '')} |")

        lines.extend(["", "## 整改行动计划", ""])
        for index, rec in enumerate(result.get("recommendations", []), start=1):
            lines.extend(
                [
                    f"### {index}. {rec.get('type', '')}（{rec.get('priority', '')}）",
                    "",
                    rec.get("description", ""),
                    "",
                    f"- 责任角色：{rec.get('owner_role', '')}",
                    f"- 期限：{rec.get('due_days', '')} 天",
                    f"- 验收指标：{rec.get('success_metric', '')}",
                    f"- 动作：{'；'.join(rec.get('action_items', []) or [])}",
                    "",
                ]
            )

        lines.extend(["", "## 整改任务跟踪", "", "| 任务 | 状态 | 责任人 | 到期天数 | 验收指标 |", "| --- | --- | --- | --- | --- |"])
        for task in record.get("remediation_tasks", []):
            lines.append(
                f"| {task.get('task_id', '')} {task.get('title', '')} | {task.get('status', '')} | "
                f"{task.get('owner', '')} | {task.get('due_days', '')} | {self._clean_table(task.get('success_metric', ''))} |"
            )

        lines.extend(["", "## 复核记录", ""])
        reviews = record.get("reviews", [])
        if not reviews:
            lines.append("暂无复核记录。")
        for review in reviews:
            lines.append(f"- {review.get('created_at')} / {review.get('reviewer')} / {review.get('decision')}：{review.get('comment')}")

        return "\n".join(lines)

    def _write(self, record: Dict[str, Any]) -> None:
        record["updated_at"] = datetime.now().isoformat()
        self._path(record["run_id"]).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _build_remediation_tasks(self, run_id: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        tasks = []
        for index, rec in enumerate(result.get("recommendations", []), start=1):
            tasks.append(
                {
                    "task_id": f"{run_id}-TASK-{index:02d}",
                    "title": rec.get("type", "整改任务"),
                    "description": rec.get("description", ""),
                    "priority": rec.get("priority", "中"),
                    "owner": rec.get("owner_role", "控制责任人"),
                    "due_days": rec.get("due_days", 30),
                    "success_metric": rec.get("success_metric", ""),
                    "action_items": rec.get("action_items", []),
                    "status": "未开始",
                    "created_at": datetime.now().isoformat(),
                    "updated_at": datetime.now().isoformat(),
                    "notes": [],
                }
            )
        return tasks

    def _build_evidence_requests(self, run_id: str, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        requests = []
        seen = set()
        for control in result.get("control_matrix", []):
            for evidence in control.get("evidence_required", []):
                if evidence in seen:
                    continue
                seen.add(evidence)
                requests.append(
                    {
                        "request_id": f"{run_id}-EV-{len(requests) + 1:02d}",
                        "evidence": evidence,
                        "source": "现场取证",
                        "usage": f"验证 {control.get('control_id')} {control.get('domain')} 控制",
                        "owner": "控制责任人",
                        "status": "待收集",
                        "priority": "高" if evidence in result.get("quality_gate", {}).get("missing_evidence", []) else "中",
                        "created_at": datetime.now().isoformat(),
                        "updated_at": datetime.now().isoformat(),
                        "notes": [],
                    }
                )
        return requests

    def _build_control_tests(self, result: Dict[str, Any]) -> List[Dict[str, Any]]:
        procedure_by_control = {item.get("control_id"): item for item in result.get("audit_program", [])}
        tests = []
        for control in result.get("control_matrix", []):
            procedure = procedure_by_control.get(control.get("control_id"), {})
            tests.append(
                {
                    "control_id": control.get("control_id"),
                    "domain": control.get("domain"),
                    "assertion": procedure.get("assertion"),
                    "procedure": procedure.get("procedure") or control.get("test_procedure"),
                    "workpaper_ref": procedure.get("workpaper_ref"),
                    "sample_method": procedure.get("method"),
                    "result": "待执行",
                    "tester": "审计员",
                    "exceptions": [],
                    "updated_at": datetime.now().isoformat(),
                }
            )
        return tests

    def _normalize_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record["status"] = self._clean_legacy(record.get("status", "待复核"))
        record["lifecycle_stage"] = self._clean_legacy(record.get("lifecycle_stage", "取证"))
        request = record.setdefault("request", {})
        request["audit_item"] = self._display_text(request.get("audit_item"), "历史审计档案")
        request["audit_type"] = self._display_text(request.get("audit_type"), "综合审计")
        request["risk_level"] = self._clean_legacy(request.get("risk_level", "中"))
        result = record.get("result", {})
        if isinstance(result.get("response"), str) and self._looks_corrupt(result.get("response")):
            result["response"] = "该历史档案由旧版本生成，原始文本存在编码污染。请重新运行审计以生成完整中文报告；历史 JSON 已保留用于追溯。"
        risk = result.get("risk_assessment", {})
        if risk.get("risk_level"):
            risk["risk_level"] = self._clean_legacy(risk.get("risk_level"))
        for task in record.get("remediation_tasks", []):
            task["status"] = self._clean_legacy(task.get("status", "未开始"))
        if "evidence_requests" not in record:
            record["evidence_requests"] = self._build_evidence_requests(record.get("run_id", "AR"), result)
        if "control_tests" not in record:
            record["control_tests"] = self._build_control_tests(result)
        record.setdefault("evidence_analyses", [])
        record.setdefault("events", [])
        return record

    def _append_event(self, record: Dict[str, Any], event_type: str, message: str) -> None:
        record.setdefault("events", []).append({"at": datetime.now().isoformat(), "type": event_type, "message": message})

    def _clean_legacy(self, value: Any) -> str:
        text = str(value or "")
        if "\u6942" in text:
            return "高"
        if "\u6d93" in text:
            return "中"
        if "\u6d63" in text:
            return "低"
        mapping = {
            "todo": "未开始",
            "doing": "进行中",
            "verifying": "待验证",
            "done": "已完成",
            "closed": "已关闭",
            "pass": "通过",
            "review": "需复核",
            "blocked": "阻塞",
        }
        if text in mapping:
            return mapping[text]
        if any(mark in text for mark in ["\u5bf0", "\u5bb8", "\u93c1", "\u7487", "\u20ac", "\ufffd"]):
            return "待复核"
        return text or "未知"

    def _looks_corrupt(self, value: Any) -> bool:
        text = str(value or "")
        return ("\u003f" * 3) in text or any(mark in text for mark in ["\u93c1", "\u7487", "\u20ac", "\ufffd"])

    def _display_text(self, value: Any, fallback: str) -> str:
        text = str(value or "").strip()
        if not text or self._looks_corrupt(text):
            return fallback
        return text

    def _path(self, run_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
        return self.root / f"{safe}.json"

    def _clean_table(self, text: str) -> str:
        return str(text).replace("|", "/").replace("\n", " ")

    def _parse_date(self, value: str | None) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None

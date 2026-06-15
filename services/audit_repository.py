"""Persistent audit run storage and report rendering."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS


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
            "status": "待复核" if result.get("quality_gate", {}).get("escalation_required") else "已生成",
            "request": request,
            "result": result,
            "remediation_tasks": self._build_remediation_tasks(run_id, result),
            "reviews": [],
        }
        self._write(record)
        return record

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        records = []
        for path in self.root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                result = record.get("result", {})
                records.append(
                    {
                        "run_id": record.get("run_id"),
                        "created_at": record.get("created_at"),
                        "updated_at": record.get("updated_at"),
                        "status": record.get("status"),
                        "audit_item": record.get("request", {}).get("audit_item"),
                        "audit_type": record.get("request", {}).get("audit_type"),
                        "risk_level": result.get("risk_assessment", {}).get("risk_level"),
                        "risk_score": result.get("risk_assessment", {}).get("risk_score"),
                        "quality_confidence": result.get("quality_gate", {}).get("confidence"),
                        "compliance_score": result.get("compliance_check", {}).get("compliance_score"),
                    }
                )
            except Exception:
                continue
        records.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return records[:limit]

    def iter_records(self, limit: int = 200) -> List[Dict[str, Any]]:
        records = []
        for path in self.root.glob("*.json"):
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                continue
        records.sort(key=lambda item: item.get("created_at") or "", reverse=True)
        return records[:limit]

    def task_summary(self) -> Dict[str, Any]:
        status_counts: Dict[str, int] = {}
        open_tasks = 0
        overdue_tasks = 0
        for path in self.root.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            created_at = self._parse_date(record.get("created_at"))
            for task in record.get("remediation_tasks", []):
                status = task.get("status", "未知")
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
        return json.loads(path.read_text(encoding="utf-8"))

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
        record["status"] = "已通过" if decision == "approve" else "需整改" if decision == "reject" else "待补证"
        record["updated_at"] = datetime.now().isoformat()
        self._write(record)
        return record

    def update_task(self, run_id: str, task_id: str, status: str, owner: str = "", note: str = "") -> Optional[Dict[str, Any]]:
        record = self.get_run(run_id)
        if not record:
            return None
        for task in record.get("remediation_tasks", []):
            if task.get("task_id") == task_id:
                task["status"] = status
                if owner:
                    task["owner"] = owner
                if note:
                    task.setdefault("notes", []).append({"note": note, "at": datetime.now().isoformat()})
                task["updated_at"] = datetime.now().isoformat()
                record["updated_at"] = datetime.now().isoformat()
                self._write(record)
                return record
        return None

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
            f"# 智能审计报告 - {run_id}",
            "",
            f"- 审计对象：{request.get('audit_item', '')}",
            f"- 审计类型：{request.get('audit_type', '')}",
            f"- 参考标准：{request.get('standard_type', '')}",
            f"- 生成时间：{record.get('created_at', '')}",
            f"- 当前状态：{record.get('status', '')}",
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

        lines.extend(["## 复核记录", ""])
        reviews = record.get("reviews", [])
        if not reviews:
            lines.append("暂无复核记录。")
        for review in reviews:
            lines.append(f"- {review.get('created_at')} / {review.get('reviewer')} / {review.get('decision')}：{review.get('comment')}")

        return "\n".join(lines)

    def _write(self, record: Dict[str, Any]) -> None:
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

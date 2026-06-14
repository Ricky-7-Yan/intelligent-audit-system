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

        lines.extend(["## 复核记录", ""])
        reviews = record.get("reviews", [])
        if not reviews:
            lines.append("暂无复核记录。")
        for review in reviews:
            lines.append(f"- {review.get('created_at')} / {review.get('reviewer')} / {review.get('decision')}：{review.get('comment')}")

        return "\n".join(lines)

    def _write(self, record: Dict[str, Any]) -> None:
        self._path(record["run_id"]).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")

    def _path(self, run_id: str) -> Path:
        safe = re.sub(r"[^A-Za-z0-9_-]", "_", run_id)
        return self.root / f"{safe}.json"

    def _clean_table(self, text: str) -> str:
        return str(text).replace("|", "/").replace("\n", " ")

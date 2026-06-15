"""Product-facing metrics for the audit delivery workspace."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
from typing import Any, Dict, List

from config import RAG_CONFIG
from services.audit_repository import AuditRunRepository
from services.skill_registry import SkillRegistry


class ProductInsights:
    def __init__(self, audit_repository: AuditRunRepository, skill_registry: SkillRegistry) -> None:
        self.audit_repository = audit_repository
        self.skill_registry = skill_registry

    def overview(self, rag_stats: Dict[str, Any] | None = None) -> Dict[str, Any]:
        rag_stats = self._rag_stats(rag_stats)
        runs = self.audit_repository.list_runs(limit=200)
        risk_counter = Counter(run.get("risk_level") or "未评估" for run in runs)
        status_counter = Counter(run.get("status") or "未知" for run in runs)
        task_summary = self.audit_repository.task_summary()
        summary = {
            "audit_runs": len(runs),
            "open_tasks": task_summary["open_tasks"],
            "overdue_tasks": task_summary["overdue_tasks"],
            "avg_quality": self._avg([run.get("quality_confidence") for run in runs]),
            "avg_compliance": self._avg([run.get("compliance_score") for run in runs]),
            "knowledge_chunks": rag_stats.get("total_documents", 0),
            "skills": len(self.skill_registry.list_skills()),
        }
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": summary,
            "risk_distribution": dict(risk_counter),
            "status_distribution": dict(status_counter),
            "task_status": task_summary["status_distribution"],
            "recent_runs": runs[:8],
            "recent_skill_runs": self.skill_registry.recent_runs(8),
            "risk_register": self.risk_register(runs),
            "evidence_requests": self.evidence_requests(),
            "control_health": self.control_health(runs),
            "connectors": self._connectors(rag_stats),
            "pipeline": self._pipeline(),
            "customer_value": self._customer_value(),
        }

    def risk_register(self, runs: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        runs = runs if runs is not None else self.audit_repository.list_runs(limit=200)
        register = []
        for run in runs[:20]:
            register.append(
                {
                    "risk_id": f"RR-{run.get('run_id', '')[-6:]}",
                    "audit_item": run.get("audit_item") or "待审计对象",
                    "risk_level": run.get("risk_level") or "未评估",
                    "risk_score": run.get("risk_score") or 0,
                    "owner": "审计经理",
                    "status": run.get("status") or "待处理",
                    "next_action": self._next_action(run),
                }
            )
        return register

    def evidence_requests(self) -> List[Dict[str, Any]]:
        requests: List[Dict[str, Any]] = []
        for record in self.audit_repository.iter_records(limit=80):
            for item in record.get("evidence_requests", [])[:8]:
                requests.append(
                    {
                        "request_id": item.get("request_id"),
                        "audit_item": record.get("request", {}).get("audit_item") or "待审计对象",
                        "evidence": item.get("evidence"),
                        "owner": item.get("owner") or "控制责任人",
                        "priority": item.get("priority") or "中",
                        "status": item.get("status") or "待收集",
                    }
                )
        return requests[:24]

    def control_health(self, runs: List[Dict[str, Any]] | None = None) -> List[Dict[str, Any]]:
        records = self.audit_repository.iter_records(limit=80)
        domain_stats: Dict[str, Dict[str, Any]] = {}
        for record in records:
            for control in record.get("result", {}).get("control_matrix", []):
                domain = control.get("domain") or "通用控制"
                stats = domain_stats.setdefault(domain, {"domain": domain, "controls": 0, "maturity_sum": 0.0, "exceptions": 0})
                stats["controls"] += 1
                stats["maturity_sum"] += float(control.get("maturity_level") or 0)
                if "取证" in str(control.get("status", "")) or "验证" in str(control.get("status", "")):
                    stats["exceptions"] += 1
        health = []
        for stats in domain_stats.values():
            controls = max(stats["controls"], 1)
            maturity = round(stats["maturity_sum"] / controls, 2)
            health.append(
                {
                    "domain": stats["domain"],
                    "controls": stats["controls"],
                    "avg_maturity": maturity,
                    "exceptions": stats["exceptions"],
                    "health_score": round(min(100, maturity * 18 + max(0, 5 - stats["exceptions"]) * 2), 1),
                }
            )
        health.sort(key=lambda item: item["health_score"])
        return health[:12]

    def _connectors(self, rag_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"name": "Knowledge Base", "status": "online", "detail": f"{rag_stats.get('total_documents', 0)} chunks"},
            {"name": "Audit Archive", "status": "online", "detail": "local JSON repository"},
            {"name": "Skill Registry", "status": "online", "detail": f"{len(self.skill_registry.list_skills())} tools"},
            {"name": "LLM Gateway", "status": "configured", "detail": "DeepSeek/OpenAI compatible"},
            {"name": "MySQL Standards", "status": "optional", "detail": "falls back to built-in controls"},
            {"name": "Neo4j Graph", "status": "optional", "detail": "falls back to local graph"},
        ]

    def _pipeline(self) -> List[Dict[str, Any]]:
        return [
            {"stage": "Scope", "title": "范围规划", "detail": "识别系统边界、标准、风险主题和审计目标。"},
            {"stage": "Retrieve", "title": "证据检索", "detail": "从企业知识库和种子标准中召回可引用材料。"},
            {"stage": "Map", "title": "控制映射", "detail": "把风险主题映射到控制矩阵和测试程序。"},
            {"stage": "Assess", "title": "风险评分", "detail": "结合证据质量、控制成熟度和风险关键词计算剩余风险。"},
            {"stage": "Gate", "title": "质量门", "detail": "输出置信度、缺失证据和人工复核条件。"},
            {"stage": "Close", "title": "整改闭环", "detail": "生成任务、状态流转、复核记录和报告。"},
        ]

    def _customer_value(self) -> List[Dict[str, str]]:
        return [
            {"title": "审计自动化", "detail": "从审计对象直接生成范围、控制矩阵、证据包、程序和报告。"},
            {"title": "证据可追溯", "detail": "RAG 答案返回来源，质量门输出置信度和缺失证据。"},
            {"title": "整改闭环", "detail": "发现、建议、责任人、状态和复核意见保存在审计档案中。"},
            {"title": "平台化扩展", "detail": "Skill/MCP 风格工具可注册、可描述、可审计。"},
        ]

    def _next_action(self, run: Dict[str, Any]) -> str:
        if (run.get("quality_confidence") or 0) < 0.7:
            return "补充证据并提交复核"
        if run.get("risk_level") in {"高", "中"}:
            return "确认整改责任人与到期时间"
        return "归档并纳入持续监控"

    def _avg(self, values: List[Any]) -> float:
        numbers = [float(value) for value in values if isinstance(value, (int, float))]
        return round(sum(numbers) / len(numbers), 2) if numbers else 0.0

    def _rag_stats(self, rag_stats: Dict[str, Any] | None) -> Dict[str, Any]:
        if rag_stats and rag_stats.get("total_documents"):
            return rag_stats
        store_file = RAG_CONFIG["store_file"]
        try:
            payload = json.loads(store_file.read_text(encoding="utf-8"))
            return {"total_documents": len(payload.get("chunks", [])), "store_file": str(store_file)}
        except Exception:
            return rag_stats or {"total_documents": 0}

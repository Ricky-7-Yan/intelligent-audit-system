"""Product-facing metrics for the audit operations console."""

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
        avg_quality = self._avg([run.get("quality_confidence") for run in runs])
        avg_compliance = self._avg([run.get("compliance_score") for run in runs])
        open_tasks = self.audit_repository.task_summary()
        recent_runs = runs[:8]
        skill_runs = self.skill_registry.recent_runs(8)
        return {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "audit_runs": len(runs),
                "open_tasks": open_tasks["open_tasks"],
                "overdue_tasks": open_tasks["overdue_tasks"],
                "avg_quality": avg_quality,
                "avg_compliance": avg_compliance,
                "knowledge_chunks": rag_stats.get("total_documents", 0),
                "skills": len(self.skill_registry.list_skills()),
            },
            "risk_distribution": dict(risk_counter),
            "status_distribution": dict(status_counter),
            "task_status": open_tasks["status_distribution"],
            "recent_runs": recent_runs,
            "recent_skill_runs": skill_runs,
            "connectors": self._connectors(rag_stats),
            "pipeline": self._pipeline(),
            "customer_value": [
                {"title": "审计自动化", "detail": "从审计对象直接生成范围、控制矩阵、证据包、程序和报告。"},
                {"title": "证据可追溯", "detail": "RAG 答案返回来源，质量门输出置信度和缺失证据。"},
                {"title": "整改闭环", "detail": "发现、建议、责任人、状态和复核意见保存在审计档案中。"},
                {"title": "平台化扩展", "detail": "Skill/MCP 风格工具可注册、可描述、可审计。"},
            ],
        }

    def _connectors(self, rag_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"name": "Knowledge Base", "status": "online", "detail": f"{rag_stats.get('total_documents', 0)} chunks"},
            {"name": "Audit Archive", "status": "online", "detail": "local JSON repository"},
            {"name": "Skill Registry", "status": "online", "detail": f"{len(self.skill_registry.list_skills())} tools"},
            {"name": "LLM Gateway", "status": "configured", "detail": "optional Qwen/OpenAI compatible"},
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

"""Product-facing metrics for the audit delivery workspace."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
import json
import statistics
from typing import Any, Dict, List

from config import RAG_CONFIG
from services.audit_repository import AuditRunRepository
from services.audit_templates import list_audit_templates
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
        scenario_coverage = self._scenario_coverage()
        summary = {
            "audit_runs": len(runs),
            "open_tasks": task_summary["open_tasks"],
            "overdue_tasks": task_summary["overdue_tasks"],
            "avg_quality": self._avg([run.get("quality_confidence") for run in runs]),
            "avg_compliance": self._avg([run.get("compliance_score") for run in runs]),
            "knowledge_chunks": rag_stats.get("total_documents", 0),
            "skills": len(self.skill_registry.list_skills()),
            "scenario_templates": scenario_coverage["built_in_templates"],
            "standards": scenario_coverage["standards_count"],
            "control_themes": scenario_coverage["control_themes_count"],
            "evidence_types": scenario_coverage["evidence_types_count"],
            "deliverable_types": scenario_coverage["deliverable_types_count"],
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
            "scenario_coverage": scenario_coverage,
            "value_measurement": self.value_baseline(),
        }

    def value_baseline(self) -> Dict[str, Any]:
        """Calculate evidence-backed pilot KPIs without inventing missing values."""
        all_records = self.audit_repository.iter_records(limit=1000)
        records = [
            record
            for record in all_records
            if str(record.get("request", {}).get("measurement_cohort") or "").lower() == "pilot"
        ]
        evidence_cycles: List[float] = []
        evidence_total = 0
        evidence_verified = 0
        reviewed_projects = 0
        first_pass_projects = 0
        completed_tasks = 0
        on_time_tasks = 0
        observed_dates: List[datetime] = []

        for record in records:
            created_at = self._parse_datetime(record.get("created_at"))
            updated_at = self._parse_datetime(record.get("updated_at"))
            observed_dates.extend(value for value in (created_at, updated_at) if value is not None)

            reviews = sorted(
                record.get("reviews", []),
                key=lambda item: str(item.get("created_at") or ""),
            )
            if reviews:
                reviewed_projects += 1
                if reviews[0].get("decision") == "approve":
                    first_pass_projects += 1

            for item in record.get("evidence_requests", []):
                status = self._clean_status(item.get("status"))
                if status == "不适用":
                    continue
                evidence_total += 1
                if status == "已验证":
                    evidence_verified += 1
                    started = self._parse_datetime(item.get("created_at")) or created_at
                    finished = self._parse_datetime(item.get("updated_at"))
                    if started and finished and finished >= started:
                        evidence_cycles.append((finished - started).total_seconds() / 3600)

            for task in record.get("remediation_tasks", []):
                status = self._clean_status(task.get("status"))
                if status not in {"已完成", "已关闭"}:
                    continue
                started = self._parse_datetime(task.get("created_at")) or created_at
                finished = self._parse_datetime(task.get("updated_at"))
                due_days = self._safe_int(task.get("due_days"), default=-1)
                if not started or not finished or due_days < 0:
                    continue
                completed_tasks += 1
                if (finished - started).total_seconds() <= due_days * 86400:
                    on_time_tasks += 1

        metrics = [
            self._duration_metric(
                "evidence_cycle_time",
                "证据满足周期",
                evidence_cycles,
                minimum_sample=10,
                definition="证据请求创建至状态更新为“已验证”的中位时长",
                direction="越低越好",
            ),
            self._ratio_metric(
                "first_pass_review_rate",
                "底稿首次复核通过率",
                first_pass_projects,
                reviewed_projects,
                minimum_sample=5,
                definition="首次人工复核结论为通过的项目占已复核项目比例",
                direction="越高越好",
            ),
            self._ratio_metric(
                "evidence_sufficiency_rate",
                "证据充分率",
                evidence_verified,
                evidence_total,
                minimum_sample=20,
                definition="状态为“已验证”的适用证据请求占全部适用证据请求比例",
                direction="越高越好",
            ),
            self._ratio_metric(
                "on_time_remediation_rate",
                "整改按期关闭率",
                on_time_tasks,
                completed_tasks,
                minimum_sample=10,
                definition="在任务创建时间加整改期限之前完成或关闭的任务比例",
                direction="越高越好",
            ),
        ]
        ready_count = sum(1 for metric in metrics if metric["status"] == "ready")
        return {
            "status": "ready" if ready_count == len(metrics) else "collecting",
            "ready_metrics": ready_count,
            "total_metrics": len(metrics),
            "projects_observed": len(records),
            "projects_available": len(all_records),
            "projects_excluded": len(all_records) - len(records),
            "measurement_scope": "pilot_only",
            "period_start": min(observed_dates).isoformat() if observed_dates else None,
            "period_end": max(observed_dates).isoformat() if observed_dates else None,
            "metrics": metrics,
            "methodology": [
                "只统计当前租户和当前用户有权访问的审计项目。",
                "只有立项时显式选择“正式试点”的项目进入价值基线；历史演示与日常项目不进入客户价值声明。",
                "未达到最小样本量时显示“待积累”，同时保留观察值和分子分母供审计。",
                "证据完成以“已验证”为准；不适用请求不进入分母。",
                "整改完成时间使用任务最后更新时间，生产连接器接入后应改用独立 completed_at。",
            ],
            "claim_policy": "这些指标是当前工作区运行基线，不等于客户节省工时或投资回报。",
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
            {"name": "Audit Archive", "status": "online", "detail": "tenant-scoped SQLite WAL"},
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
            {
                "pain": "资料散落，取证反复追问",
                "solution": "按审计范围生成证据请求，统一检索、来源定位和缺口提示。",
                "proof": "证据请求单、页段级引用、缺失证据清单",
                "audience": "审计员 / 业务责任人",
            },
            {
                "pain": "控制测试依赖个人经验，口径不一致",
                "solution": "用版本化场景模板把风险、控制、证据和测试程序映射到同一工作底稿。",
                "proof": "控制矩阵、抽样计划、测试程序、例外清单",
                "audience": "项目经理 / 审计员",
            },
            {
                "pain": "AI 结论难复核，无法直接进入底稿",
                "solution": "保存执行轨迹、工具调用和证据链，低置信或缺证结果自动进入人工复核。",
                "proof": "质量门、来源引用、复核记录、不可变交付物",
                "audience": "复核人 / 审计经理",
            },
            {
                "pain": "发现和整改脱节，复核容易失去上下文",
                "solution": "把发现、建议、责任人、到期日、状态流转和复核意见保存在同一审计档案。",
                "proof": "整改任务、逾期提示、复核意见、交付报告",
                "audience": "整改责任人 / 管理层",
            },
        ]

    def _scenario_coverage(self) -> Dict[str, Any]:
        templates = list_audit_templates()
        standards = sorted({item["standard"] for item in templates})
        control_themes = sorted({value for item in templates for value in item.get("scope", [])})
        evidence_types = sorted({value for item in templates for value in item.get("evidence", [])})
        deliverable_types = sorted({value for item in templates for value in item.get("deliverables", [])})
        pain_by_template = {
            "tpl-itgc-sox": "财务系统关键控制多、样本与证据难统一",
            "tpl-erp-access": "越权与职责冲突难以从账号、角色和审批中快速定位",
            "tpl-data-security": "数据目录、访问、共享和日志分散，合规缺口难串联",
            "tpl-change-release": "变更证据跨需求、测试、审批与上线环节，例外易漏检",
            "tpl-backup-recovery": "有备份不等于可恢复，RPO/RTO 与演练结果难闭环",
            "tpl-third-party": "供应商准入、合同、访问和退出责任分散",
        }
        scenarios = []
        for item in templates:
            scenarios.append(
                {
                    "template_id": item["template_id"],
                    "name": item["name"],
                    "audit_type": item["audit_type"],
                    "standard": item["standard"],
                    "risk_level": item["risk_level"],
                    "pain": pain_by_template[item["template_id"]],
                    "scope_count": len(item.get("scope", [])),
                    "evidence_count": len(item.get("evidence", [])),
                    "deliverable_count": len(item.get("deliverables", [])),
                    "example_deliverables": item.get("deliverables", [])[:2],
                }
            )
        return {
            "built_in_templates": len(templates),
            "custom_scenarios_supported": True,
            "standards": standards,
            "standards_count": len(standards),
            "control_themes_count": len(control_themes),
            "evidence_types_count": len(evidence_types),
            "deliverable_types_count": len(deliverable_types),
            "scenarios": scenarios,
            "coverage_statement": "内置 6 个高频数字化审计模板，并支持按企业控制库扩展自定义场景。",
            "methodology": "数量来自版本化审计模板的去重统计，不代表覆盖全部行业和全部审计业务。",
        }

    def value_baseline_csv(self) -> str:
        baseline = self.value_baseline()
        lines = ["metric_id,metric,display_value,observed_value,numerator,denominator,sample_size,minimum_sample,status,direction,definition"]
        for item in baseline["metrics"]:
            values = [
                item.get("metric_id"), item.get("metric"), item.get("display_value"), item.get("observed_value"),
                item.get("numerator"), item.get("denominator"), item.get("sample_size"), item.get("minimum_sample"),
                item.get("status"), item.get("direction"), item.get("definition"),
            ]
            lines.append(",".join(self._csv_cell(value) for value in values))
        return "\ufeff" + "\n".join(lines) + "\n"

    def _ratio_metric(
        self,
        metric_id: str,
        metric: str,
        numerator: int,
        denominator: int,
        minimum_sample: int,
        definition: str,
        direction: str,
    ) -> Dict[str, Any]:
        observed = round(numerator / denominator * 100, 1) if denominator else None
        ready = denominator >= minimum_sample
        return {
            "metric_id": metric_id,
            "metric": metric,
            "display_value": f"{observed:.1f}%" if ready and observed is not None else "待积累",
            "observed_value": observed,
            "unit": "%",
            "numerator": numerator,
            "denominator": denominator,
            "sample_size": denominator,
            "minimum_sample": minimum_sample,
            "status": "ready" if ready else "insufficient_sample",
            "definition": definition,
            "direction": direction,
        }

    def _duration_metric(
        self,
        metric_id: str,
        metric: str,
        values_hours: List[float],
        minimum_sample: int,
        definition: str,
        direction: str,
    ) -> Dict[str, Any]:
        observed = round(statistics.median(values_hours), 1) if values_hours else None
        ready = len(values_hours) >= minimum_sample
        return {
            "metric_id": metric_id,
            "metric": metric,
            "display_value": self._format_hours(observed) if ready and observed is not None else "待积累",
            "observed_value": observed,
            "unit": "hours",
            "numerator": None,
            "denominator": None,
            "sample_size": len(values_hours),
            "minimum_sample": minimum_sample,
            "status": "ready" if ready else "insufficient_sample",
            "definition": definition,
            "direction": direction,
        }

    def _format_hours(self, hours: float) -> str:
        if hours >= 48:
            return f"{hours / 24:.1f} 天"
        return f"{hours:.1f} 小时"

    def _parse_datetime(self, value: Any) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
        except ValueError:
            return None

    def _safe_int(self, value: Any, default: int = 0) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    def _clean_status(self, value: Any) -> str:
        mapping = {"done": "已完成", "closed": "已关闭", "verified": "已验证"}
        text = str(value or "").strip()
        return mapping.get(text.lower(), text)

    def _csv_cell(self, value: Any) -> str:
        text = "" if value is None else str(value)
        return '"' + text.replace('"', '""') + '"'

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

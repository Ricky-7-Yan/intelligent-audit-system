"""
Training data helpers and deterministic agent benchmark evaluation.

Heavy SFT/RLHF/RLVR jobs should run offline. The web process keeps only a
lightweight, reproducible evaluation layer for audit-domain Agent quality.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS, TRAINING_CONFIG


logger = logging.getLogger(__name__)


@dataclass
class TrainingData:
    instruction: str
    input: str
    output: str
    category: str
    difficulty: str
    source: str


class DataCollector:
    def __init__(self) -> None:
        self.collected_data: List[TrainingData] = []

    def collect_audit_standards_data(self) -> List[TrainingData]:
        return [
            TrainingData(
                instruction="解释审计标准",
                input="COBIT 2019 的审计关注点是什么？",
                output="COBIT 2019 关注治理目标、价值交付、风险优化、资源优化、流程责任、绩效度量和控制活动可追溯性。",
                category="governance",
                difficulty="basic",
                source="builtin",
            ),
            TrainingData(
                instruction="解释审计标准",
                input="ISO 27001 审计应检查哪些证据？",
                output="应检查资产清单、风险评估、控制适用性声明、访问复核、事件记录、备份演练、供应商管理和管理评审记录。",
                category="security",
                difficulty="basic",
                source="builtin",
            ),
        ]

    def collect_risk_assessment_data(self) -> List[TrainingData]:
        return [
            TrainingData(
                instruction="进行风险评估",
                input="评估 ERP 权限管理风险",
                output="重点检查最小权限、职责分离、账号生命周期、特权账号审批、定期复核、异常登录监控和权限矩阵留痕。",
                category="risk_assessment",
                difficulty="intermediate",
                source="builtin",
            )
        ]

    def collect_compliance_check_data(self) -> List[TrainingData]:
        return [
            TrainingData(
                instruction="进行合规检查",
                input="SOX 对财务系统变更管理有什么要求？",
                output="应验证变更申请、审批、测试、上线授权、回退计划、日志留存、财务报告影响评估和抽样底稿。",
                category="compliance",
                difficulty="intermediate",
                source="builtin",
            )
        ]

    def collect_all_data(self) -> List[TrainingData]:
        self.collected_data = [
            *self.collect_audit_standards_data(),
            *self.collect_risk_assessment_data(),
            *self.collect_compliance_check_data(),
        ]
        return self.collected_data

    def save_data(self, data: List[TrainingData], file_path: str) -> None:
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([asdict(item) for item in data], ensure_ascii=False, indent=2), encoding="utf-8")


class SFTTrainer:
    """Placeholder for offline SFT jobs."""

    def __init__(self, model_name: str = "deepseek-chat") -> None:
        self.model_name = model_name

    def setup_model(self):
        raise RuntimeError("SFT training should be run as an offline job, not from the web process.")

    def train(self, *args, **kwargs):
        raise RuntimeError("SFT training should be run as an offline job, not from the web process.")


class RLHFTrainer:
    """Placeholder for offline RLHF/RLVR jobs."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def setup_reward_model(self):
        raise RuntimeError("RLHF training should be run as an offline job, not from the web process.")

    def train_reward_model(self, *args, **kwargs):
        raise RuntimeError("RLHF training should be run as an offline job, not from the web process.")

    def train_with_ppo(self, *args, **kwargs):
        raise RuntimeError("PPO/RLVR training should be run as an offline job, not from the web process.")


class BenchmarkEvaluator:
    """Audit Agent evaluation aligned with enterprise Agent JD requirements."""

    DEFAULT_METRICS = [
        "faithfulness",
        "completeness",
        "audit_professionalism",
        "actionability",
        "compliance_alignment",
        "agentic_capability",
        "tool_trace_quality",
        "human_review_awareness",
    ]

    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def create_test_cases(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "AP-PLN-001",
                "category": "audit_scope_planning",
                "question": "请为 ERP 权限管理审计设计范围、关键风险、取证清单和控制测试步骤。",
                "expected_answer": "审计范围 账号生命周期 角色权限 职责分离 特权账号 定期复核 证据清单 控制测试 抽样 复核",
                "expected_terms": ["范围", "账号生命周期", "职责分离", "特权账号", "复核", "证据", "控制测试"],
                "evaluation_criteria": self.DEFAULT_METRICS,
            },
            {
                "id": "AP-RAG-002",
                "category": "rag_grounding",
                "question": "ISO27001 访问控制审计应如何证明结论有依据？",
                "expected_answer": "引用制度、资产清单、访问矩阵、审批记录、日志、复核记录和控制适用性声明，标注证据缺口。",
                "expected_terms": ["依据", "制度", "访问矩阵", "审批", "日志", "复核", "证据缺口"],
                "evaluation_criteria": self.DEFAULT_METRICS,
            },
            {
                "id": "AP-CTL-003",
                "category": "control_mapping",
                "question": "SOX ITGC 变更管理审计需要映射哪些控制和测试底稿？",
                "expected_answer": "变更申请、审批、开发测试、上线授权、职责分离、回退计划、日志留存、样本测试和异常跟踪。",
                "expected_terms": ["变更", "审批", "测试", "上线", "回退", "底稿", "异常"],
                "evaluation_criteria": self.DEFAULT_METRICS,
            },
            {
                "id": "AP-EVD-004",
                "category": "evidence_request",
                "question": "发现财务系统存在离职账号未禁用，应如何补充审计证据并推进整改？",
                "expected_answer": "补充人员离职清单、账号状态、禁用时间、审批记录、影响范围、整改责任人、复核计划和关闭标准。",
                "expected_terms": ["离职", "账号", "禁用", "影响范围", "整改", "责任人", "关闭标准"],
                "evaluation_criteria": self.DEFAULT_METRICS,
            },
            {
                "id": "AP-AGT-005",
                "category": "agentic_workflow",
                "question": "Agent 在审计中调用工具失败或证据不足时应该怎样处理？",
                "expected_answer": "记录轨迹、降级检索、重试或换源、标注不确定性、触发人工复核、生成补证任务并避免无依据结论。",
                "expected_terms": ["轨迹", "降级", "重试", "不确定", "人工复核", "补证", "依据"],
                "evaluation_criteria": self.DEFAULT_METRICS,
            },
            {
                "id": "AP-DR-006",
                "category": "deep_research",
                "question": "请跨制度、日志和访谈材料分析供应商远程访问是否存在高风险。",
                "expected_answer": "进行意图理解、查询改写、多源检索、交叉验证、风险判断、证据缺口和后续审计动作设计。",
                "expected_terms": ["多源", "查询改写", "交叉验证", "风险判断", "证据缺口", "后续动作"],
                "evaluation_criteria": self.DEFAULT_METRICS,
            },
        ]

    def evaluate_agent(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        from agents.audit_agent import AuditAgent

        agent = AuditAgent()
        started = time.perf_counter()
        try:
            results = []
            for raw_case in test_cases or self.create_test_cases():
                test_case = self._normalize_case(raw_case)
                case_started = time.perf_counter()
                output = agent.process_audit_query(test_case["question"])
                response = output.get("response", "")
                evaluation = self._evaluate_response(
                    test_case["question"],
                    response,
                    test_case.get("expected_answer", ""),
                    test_case.get("expected_terms", []),
                    test_case.get("evaluation_criteria", []),
                    output,
                )
                trajectory = self._evaluate_trajectory(output)
                latency_ms = round((time.perf_counter() - case_started) * 1000)
                results.append(
                    {
                        "test_id": test_case["id"],
                        "category": test_case["category"],
                        "question": test_case["question"],
                        "expected_answer": test_case.get("expected_answer", ""),
                        "expected_terms": test_case.get("expected_terms", []),
                        "actual_answer": response,
                        "evaluation": evaluation,
                        "trajectory": trajectory,
                        "latency_ms": latency_ms,
                        "regression_risks": self._regression_risks(evaluation, trajectory, latency_ms),
                        "optimization_suggestions": self._suggestions(evaluation, trajectory),
                    }
                )
            return {
                "results": results,
                "overall_metrics": self._calculate_overall_metrics(results),
                "evaluation_framework": {
                    "metrics": self.DEFAULT_METRICS,
                    "jd_alignment": [
                        "agent trajectory quality",
                        "tool-use precision",
                        "context and memory management",
                        "human-in-the-loop",
                        "RAG grounding",
                        "badcase regression loop",
                    ],
                },
                "training_config": TRAINING_CONFIG,
                "evaluation_date": datetime.now().isoformat(),
                "elapsed_ms": round((time.perf_counter() - started) * 1000),
            }
        finally:
            agent.close()

    def evaluate_model(self, model, tokenizer, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.evaluate_agent(test_cases)

    def _normalize_case(self, item: Dict[str, Any]) -> Dict[str, Any]:
        expected = item.get("expected_answer") or item.get("expected") or ""
        terms = item.get("expected_terms")
        if not terms and expected:
            terms = [term for term in expected.replace("、", " ").replace("，", " ").replace(",", " ").split() if term]
        return {
            "id": item.get("id") or item.get("case_id") or "custom",
            "category": item.get("category") or "custom",
            "question": item["question"],
            "expected_answer": expected,
            "expected_terms": terms or [],
            "evaluation_criteria": item.get("evaluation_criteria") or item.get("metrics") or self.DEFAULT_METRICS,
        }

    def _evaluate_response(
        self,
        question: str,
        response: str,
        expected: str,
        expected_terms: List[str],
        criteria: List[str],
        agent_output: Dict[str, Any],
    ) -> Dict[str, float]:
        source_count = len(agent_output.get("retrieved_context", {}).get("sources", [])) + len(agent_output.get("evidence_pack", []))
        quality_gate = agent_output.get("quality_gate", {})
        metrics = {
            "faithfulness": max(self._term_score(response, expected_terms), min(source_count / 6, 1.0) * 0.75),
            "completeness": min(len(response) / max(len(expected) * 2, 120), 1.0),
            "audit_professionalism": self._keyword_score(response, ["审计", "风险", "控制", "证据", "底稿", "抽样", "整改", "复核"]),
            "actionability": self._keyword_score(response, ["检查", "获取", "验证", "记录", "审批", "整改", "责任人", "时间表"]),
            "compliance_alignment": self._keyword_score(response, ["COBIT", "ISO", "SOX", "合规", "标准", "要求", "职责分离", "访问控制"]),
            "agentic_capability": self._keyword_score(response, ["任务", "步骤", "证据", "质量门", "置信度", "人工复核", "闭环", "工具"]),
            "tool_trace_quality": self._trace_score(agent_output.get("execution_trace", [])),
            "human_review_awareness": 1.0 if quality_gate.get("escalation_required") or "复核" in response or "人工" in response else 0.45,
        }
        selected = criteria or self.DEFAULT_METRICS
        return {key: round(value, 3) for key, value in metrics.items() if key in selected}

    def _evaluate_trajectory(self, output: Dict[str, Any]) -> Dict[str, Any]:
        trace = output.get("execution_trace", [])
        stages = [item.get("stage") for item in trace]
        expected_stages = ["planner", "retriever", "control_mapper", "risk_engine", "quality_gate"]
        stage_coverage = self._term_score(" ".join(str(stage) for stage in stages), expected_stages)
        quality_gate = output.get("quality_gate", {})
        return {
            "steps": len(trace),
            "stage_coverage": stage_coverage,
            "quality_gate_status": quality_gate.get("status", "unknown"),
            "confidence": quality_gate.get("confidence", 0),
            "missing_evidence_count": len(quality_gate.get("missing_evidence", [])),
            "has_human_review_trigger": bool(quality_gate.get("escalation_required")),
        }

    def _trace_score(self, trace: List[Dict[str, Any]]) -> float:
        if not trace:
            return 0.0
        expected = ["planner", "retriever", "control_mapper", "risk_engine", "quality_gate"]
        text = " ".join(str(item.get("stage", "")) for item in trace)
        return self._term_score(text, expected)

    def _term_score(self, text: str, terms: List[str]) -> float:
        if not terms:
            return 0.0
        lowered = text.lower()
        matched = sum(1 for term in terms if str(term).lower() in lowered)
        return round(matched / len(terms), 3)

    def _keyword_score(self, response: str, keywords: List[str]) -> float:
        return self._term_score(response, keywords)

    def _regression_risks(self, evaluation: Dict[str, float], trajectory: Dict[str, Any], latency_ms: int) -> List[str]:
        risks = []
        if evaluation.get("faithfulness", 1) < 0.55:
            risks.append("依据不足或关键词命中偏低，可能出现无证据结论。")
        if evaluation.get("tool_trace_quality", 1) < 0.8:
            risks.append("Agent 轨迹阶段不完整，需检查规划、检索、控制映射或质量门。")
        if trajectory.get("missing_evidence_count", 0) > 2 and not trajectory.get("has_human_review_trigger"):
            risks.append("存在多个证据缺口但未触发人工复核。")
        if latency_ms > 3000:
            risks.append("单用例耗时超过 3 秒，需关注检索或工具调用效率。")
        return risks or ["未发现明显回归风险。"]

    def _suggestions(self, evaluation: Dict[str, float], trajectory: Dict[str, Any]) -> List[str]:
        suggestions = []
        if evaluation.get("faithfulness", 1) < 0.7:
            suggestions.append("补充制度、底稿、日志和访谈样本到知识库，并要求答案引用来源。")
        if evaluation.get("actionability", 1) < 0.7:
            suggestions.append("在 Prompt/Skill 输出格式中固定责任人、证据、抽样、复核和关闭标准。")
        if evaluation.get("agentic_capability", 1) < 0.7:
            suggestions.append("增强工具选择、失败降级、Memory 摘要和 Human-in-the-loop 触发规则。")
        if trajectory.get("stage_coverage", 1) < 1:
            suggestions.append("把规划、检索、映射、风险评分、质量门作为强制运行阶段并记录 trace。")
        return suggestions or ["保持当前能力，继续通过真实审计 badcase 做回归评测。"]

    def _calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        category_scores: Dict[str, List[float]] = {}
        metric_scores: Dict[str, List[float]] = {}
        all_scores: List[float] = []
        for result in results:
            scores = list(result["evaluation"].values())
            if not scores:
                continue
            avg_score = sum(scores) / len(scores)
            all_scores.append(avg_score)
            category_scores.setdefault(result["category"], []).append(avg_score)
            for metric, score in result["evaluation"].items():
                metric_scores.setdefault(metric, []).append(score)

        return {
            "overall_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0,
            "total_tests": len(results),
            "category_scores": {
                category: round(sum(scores) / len(scores), 3)
                for category, scores in category_scores.items()
            },
            "metric_scores": {
                metric: round(sum(scores) / len(scores), 3)
                for metric, scores in metric_scores.items()
            },
            "pass_rate": round(sum(1 for score in all_scores if score >= 0.7) / len(all_scores), 3) if all_scores else 0.0,
            "avg_latency_ms": round(sum(item.get("latency_ms", 0) for item in results) / len(results)) if results else 0,
            "regression_count": sum(1 for item in results for risk in item.get("regression_risks", []) if "未发现" not in risk),
        }


if __name__ == "__main__":
    collector = DataCollector()
    data = collector.collect_all_data()
    collector.save_data(data, str(PATHS["training_data"] / "audit_training_data.json"))
    print(json.dumps(BenchmarkEvaluator().evaluate_agent(), ensure_ascii=False, indent=2))

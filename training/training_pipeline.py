"""
Training data helpers and lightweight benchmark evaluation.

Heavy SFT/RLHF training is intentionally not executed from the web process. This
module keeps a stable interface for collecting examples and evaluating the
current audit agent without loading large models.
"""

from __future__ import annotations

import json
import logging
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
                output="COBIT 2019 关注 IT 治理目标、价值交付、风险优化、资源优化、责任分工和绩效度量。",
                category="governance",
                difficulty="basic",
                source="builtin",
            ),
            TrainingData(
                instruction="解释审计标准",
                input="ISO 27001 审计应检查哪些证据？",
                output="应检查资产清单、风险评估、控制适用性声明、访问复核、事件记录、备份演练和管理评审。",
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
                output="重点检查最小权限、职责分离、账号生命周期、特权账号审批、定期复核和异常登录监控。",
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
                output="应验证变更申请、审批、测试、上线授权、回退计划、日志留存和财务报告影响评估。",
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

    def __init__(self, model_name: str = "Qwen/Qwen-7B-Chat") -> None:
        self.model_name = model_name

    def setup_model(self):
        raise RuntimeError("SFT training should be run as an offline job, not from the web process.")

    def train(self, *args, **kwargs):
        raise RuntimeError("SFT training should be run as an offline job, not from the web process.")


class RLHFTrainer:
    """Placeholder for offline RLHF jobs."""

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path

    def setup_reward_model(self):
        raise RuntimeError("RLHF training should be run as an offline job, not from the web process.")

    def train_reward_model(self, *args, **kwargs):
        raise RuntimeError("RLHF training should be run as an offline job, not from the web process.")

    def train_with_ppo(self, *args, **kwargs):
        raise RuntimeError("PPO training should be run as an offline job, not from the web process.")


class BenchmarkEvaluator:
    def __init__(self) -> None:
        self.results: List[Dict[str, Any]] = []

    def create_test_cases(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": "test_001",
                "category": "governance",
                "question": "COBIT 2019 框架的核心审计关注点是什么？",
                "expected_answer": "IT 治理、价值交付、风险优化、资源优化、流程责任和绩效度量",
                "evaluation_criteria": ["accuracy", "completeness", "professionalism"],
            },
            {
                "id": "test_002",
                "category": "risk_assessment",
                "question": "如何评估 ERP 系统权限管理风险？",
                "expected_answer": "最小权限、职责分离、特权账号、定期复核、审批证据和异常监控",
                "evaluation_criteria": ["accuracy", "practicality", "professionalism"],
            },
            {
                "id": "test_003",
                "category": "compliance",
                "question": "SOX 对财务系统内部控制审计的重点是什么？",
                "expected_answer": "职责分离、变更管理、访问控制、日志留存、财务数据完整性和管理层复核",
                "evaluation_criteria": ["accuracy", "compliance", "practicality"],
            },
            {
                "id": "test_004",
                "category": "security",
                "question": "ISO 27001 审计应关注哪些控制证据？",
                "expected_answer": "资产清单、风险评估、控制适用性声明、访问复核、事件响应和管理评审",
                "evaluation_criteria": ["accuracy", "completeness", "compliance"],
            },
        ]

    def evaluate_agent(self, test_cases: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        from agents.audit_agent import AuditAgent

        agent = AuditAgent()
        try:
            results = []
            for test_case in test_cases or self.create_test_cases():
                response = agent.process_audit_query(test_case["question"])["response"]
                evaluation = self._evaluate_response(
                    test_case["question"],
                    response,
                    test_case["expected_answer"],
                    test_case.get("evaluation_criteria", []),
                )
                results.append(
                    {
                        "test_id": test_case["id"],
                        "category": test_case["category"],
                        "question": test_case["question"],
                        "expected_answer": test_case["expected_answer"],
                        "actual_answer": response,
                        "evaluation": evaluation,
                    }
                )
            return {
                "results": results,
                "overall_metrics": self._calculate_overall_metrics(results),
                "training_config": TRAINING_CONFIG,
                "evaluation_date": datetime.now().isoformat(),
            }
        finally:
            agent.close()

    def evaluate_model(self, model, tokenizer, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        return self.evaluate_agent(test_cases)

    def _evaluate_response(self, question: str, response: str, expected: str, criteria: List[str]) -> Dict[str, float]:
        metrics = {
            "accuracy": self._calculate_overlap(response, expected),
            "completeness": min(len(response) / max(len(expected) * 2, 1), 1.0),
            "professionalism": self._keyword_score(response, ["审计", "风险", "控制", "合规", "证据", "标准", "复核", "权限"]),
            "practicality": self._keyword_score(response, ["检查", "建立", "复核", "记录", "审批", "整改", "监控", "证据"]),
            "compliance": self._keyword_score(response, ["COBIT", "ISO", "SOX", "法规", "标准", "要求", "合规", "控制"]),
        }
        return {key: value for key, value in metrics.items() if not criteria or key in criteria}

    def _calculate_overlap(self, response: str, expected: str) -> float:
        expected_terms = {term for term in expected.replace("、", " ").replace("，", " ").split() if term}
        if not expected_terms:
            return 0.0
        matched = sum(1 for term in expected_terms if term in response)
        return round(matched / len(expected_terms), 3)

    def _keyword_score(self, response: str, keywords: List[str]) -> float:
        matched = sum(1 for keyword in keywords if keyword.lower() in response.lower())
        return round(min(matched / max(len(keywords), 1), 1.0), 3)

    def _calculate_overall_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        category_scores: Dict[str, List[float]] = {}
        all_scores: List[float] = []
        for result in results:
            scores = list(result["evaluation"].values())
            if not scores:
                continue
            avg_score = sum(scores) / len(scores)
            all_scores.append(avg_score)
            category_scores.setdefault(result["category"], []).append(avg_score)

        return {
            "overall_score": round(sum(all_scores) / len(all_scores), 3) if all_scores else 0.0,
            "total_tests": len(results),
            "category_scores": {
                category: round(sum(scores) / len(scores), 3)
                for category, scores in category_scores.items()
            },
        }


if __name__ == "__main__":
    collector = DataCollector()
    data = collector.collect_all_data()
    collector.save_data(data, str(PATHS["training_data"] / "audit_training_data.json"))
    print(json.dumps(BenchmarkEvaluator().evaluate_agent(), ensure_ascii=False, indent=2))

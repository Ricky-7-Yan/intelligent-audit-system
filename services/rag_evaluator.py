"""Lightweight RAG and agent capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List


@dataclass
class RAGEvalCase:
    case_id: str
    question: str
    expected_terms: List[str]
    category: str


DEFAULT_RAG_CASES = [
    RAGEvalCase("rag_001", "ISO27001 审计应关注哪些访问控制证据？", ["访问", "权限", "复核", "风险"], "security"),
    RAGEvalCase("rag_002", "SOX 财务系统内部控制重点是什么？", ["职责分离", "变更", "财务", "复核"], "compliance"),
    RAGEvalCase("rag_003", "数据安全审计如何检查敏感数据使用？", ["分类", "授权", "加密", "审计"], "data_security"),
    RAGEvalCase("rag_004", "COBIT 如何支持 IT 治理审计？", ["治理", "目标", "风险", "绩效"], "governance"),
]


class RAGEvaluator:
    def __init__(self, rag_pipeline: Any) -> None:
        self.rag_pipeline = rag_pipeline

    def evaluate(self, cases: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        eval_cases = [self._case_from_dict(item) for item in cases] if cases else DEFAULT_RAG_CASES
        results = []
        scores = []
        for case in eval_cases:
            answer = self.rag_pipeline.query(case.question)
            text = answer.get("answer", "")
            term_score = self._term_score(text, case.expected_terms)
            source_score = min(len(answer.get("sources", [])) / 3, 1.0)
            confidence = float(answer.get("confidence", 0.0))
            overall = round(term_score * 0.45 + source_score * 0.3 + confidence * 0.25, 3)
            scores.append(overall)
            results.append(
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "question": case.question,
                    "expected_terms": case.expected_terms,
                    "term_score": term_score,
                    "source_score": round(source_score, 3),
                    "retrieval_confidence": confidence,
                    "overall": overall,
                    "retrieved_docs_count": answer.get("retrieved_docs_count", 0),
                    "sources": answer.get("sources", []),
                }
            )
        return {
            "overall_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
            "total_cases": len(results),
            "results": results,
            "evaluated_at": datetime.now().isoformat(),
        }

    def _case_from_dict(self, item: Dict[str, Any]) -> RAGEvalCase:
        return RAGEvalCase(
            case_id=item.get("case_id", "custom"),
            question=item["question"],
            expected_terms=item.get("expected_terms", []),
            category=item.get("category", "custom"),
        )

    def _term_score(self, text: str, terms: List[str]) -> float:
        if not terms:
            return 0.0
        matched = sum(1 for term in terms if term.lower() in text.lower())
        return round(matched / len(terms), 3)

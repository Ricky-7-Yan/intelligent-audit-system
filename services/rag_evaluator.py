"""Lightweight RAG and agent capability evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from services.evaluation_calibration import calibrate_continuous, mean_score


@dataclass
class RAGEvalCase:
    case_id: str
    question: str
    expected_terms: List[str]
    category: str


DEFAULT_RAG_CASES = [
    RAGEvalCase("RAG-ISO-001", "ISO27001 审计应关注哪些访问控制证据？", ["访问", "权限", "复核", "风险"], "security"),
    RAGEvalCase("RAG-SOX-002", "SOX 财务系统内部控制重点是什么？", ["职责分离", "变更", "财务", "复核"], "compliance"),
    RAGEvalCase("RAG-DATA-003", "数据安全审计如何检查敏感数据使用？", ["分类", "授权", "加密", "日志"], "data_security"),
    RAGEvalCase("RAG-COBIT-004", "COBIT 如何支持 IT 治理审计？", ["治理", "目标", "风险", "绩效"], "governance"),
    RAGEvalCase("RAG-AGENT-005", "Agent 工具调用失败时应如何记录审计轨迹？", ["工具", "轨迹", "重试", "人工复核"], "agentic_workflow"),
]


class RAGEvaluator:
    def __init__(self, rag_pipeline: Any) -> None:
        self.rag_pipeline = rag_pipeline

    def evaluate(self, cases: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
        eval_cases = [self._case_from_dict(item) for item in cases] if cases else DEFAULT_RAG_CASES
        if len(eval_cases) > 1:
            with ThreadPoolExecutor(max_workers=min(4, len(eval_cases))) as pool:
                results = list(pool.map(self._evaluate_case, eval_cases))
        else:
            results = [self._evaluate_case(case) for case in eval_cases]
        scores = [item["overall"] for item in results]
        return {
            "overall_score": mean_score(scores),
            "total_cases": len(results),
            "results": results,
            "metrics": ["term_score", "source_score", "authority_score", "retrieval_confidence"],
            "closed_loop_suggestions": self._closed_loop_suggestions(results),
            "evaluated_at": datetime.now().isoformat(),
        }

    def _evaluate_case(self, case: RAGEvalCase) -> Dict[str, Any]:
        answer = self.rag_pipeline.query(case.question)
        text = answer.get("answer", "")
        sources = answer.get("sources", [])
        term_score = calibrate_continuous(
            self._term_score(text, case.expected_terms),
            evidence_units=max(len(case.expected_terms), 1),
        )
        source_score = calibrate_continuous(
            min(len(sources) / 3, 1.0),
            evidence_units=max(len(sources), 1),
        )
        confidence = calibrate_continuous(
            float(answer.get("confidence", 0.0)),
            evidence_units=max(len(sources), 1),
        )
        authority = calibrate_continuous(
            self._authority_score(sources),
            evidence_units=max(len(sources), 1),
        )
        overall = round(term_score * 0.35 + source_score * 0.2 + confidence * 0.2 + authority * 0.25, 4)
        return {
            "case_id": case.case_id,
            "category": case.category,
            "question": case.question,
            "expected_terms": case.expected_terms,
            "term_score": term_score,
            "source_score": source_score,
            "authority_score": authority,
            "retrieval_confidence": confidence,
            "overall": overall,
            "retrieved_docs_count": answer.get("retrieved_docs_count", 0),
            "sources": sources,
            "failure_modes": self._failure_modes(term_score, source_score, authority),
        }

    def _case_from_dict(self, item: Dict[str, Any]) -> RAGEvalCase:
        terms = item.get("expected_terms") or item.get("expected") or []
        if isinstance(terms, str):
            terms = [term for term in terms.replace("、", " ").replace(",", " ").split() if term]
        return RAGEvalCase(
            case_id=item.get("case_id") or item.get("id") or "custom",
            question=item["question"],
            expected_terms=terms,
            category=item.get("category", "custom"),
        )

    def _term_score(self, text: str, terms: List[str]) -> float:
        if not terms:
            return 0.0
        lowered = text.lower()
        matched = sum(1 for term in terms if str(term).lower() in lowered)
        return round(matched / len(terms), 3)

    def _authority_score(self, sources: List[Dict[str, Any]]) -> float:
        if not sources:
            return 0.0
        trusted = 0
        for item in sources:
            source = str(item.get("source") or "").lower()
            if any(mark in source for mark in ["iso", "sox", "cobit", "builtin", "policy", "standard", "audit"]):
                trusted += 1
        return round(min(0.4 + trusted / max(len(sources), 1) * 0.6, 1.0), 3)

    def _failure_modes(self, term_score: float, source_score: float, authority: float) -> List[str]:
        failures = []
        if term_score < 0.5:
            failures.append("答案未覆盖关键审计术语或控制点。")
        if source_score < 0.5:
            failures.append("召回来源数量不足，建议补充制度、底稿或日志样本。")
        if authority < 0.6:
            failures.append("权威来源占比偏低，需优先使用标准、制度和正式底稿。")
        return failures or ["未发现明显 RAG 失败模式。"]

    def _closed_loop_suggestions(self, results: List[Dict[str, Any]]) -> List[str]:
        suggestions = []
        if any(item["term_score"] < 0.6 for item in results):
            suggestions.append("按失败用例补充审计术语同义词、控制目标和证据类型。")
        if any(item["source_score"] < 0.6 for item in results):
            suggestions.append("将企业制度、抽样底稿、系统导出和访谈纪要纳入知识库。")
        if any(item["authority_score"] < 0.6 for item in results):
            suggestions.append("为知识片段增加 source_type、owner、effective_date 和 authority_level 元数据。")
        return suggestions or ["保持当前 RAG 表现，继续纳入真实审计 badcase 做回归测试。"]

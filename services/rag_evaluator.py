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
    expected_sources: List[str] | None = None
    split: str = "validation"


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
            "metrics": [
                "term_score",
                "source_score",
                "authority_score",
                "context_relevance",
                "citation_coverage",
                "retrieval_confidence",
                "conflict_awareness",
            ],
            "dataset": {
                "origin": "user_supplied" if cases else "built_in_smoke_suite",
                "independent_test_set": bool(cases and all(item.get("split") == "test" for item in cases)),
                "warning": (
                    "内置用例仅用于冒烟与回归，不应作为业务准确率声明。"
                    if not cases
                    else "生产发布应确保测试集与开发数据独立，并保留专家标注来源。"
                ),
            },
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
        context_relevance = calibrate_continuous(
            self._context_relevance(sources),
            evidence_units=max(len(sources), 1),
        )
        citation_coverage = calibrate_continuous(
            self._citation_coverage(sources, case.expected_sources or []),
            evidence_units=max(len(sources), 1),
        )
        conflicts = answer.get("conflicts") or []
        conflict_awareness = calibrate_continuous(
            1.0 if not conflicts or answer.get("requires_human_review") else 0.0,
            evidence_units=max(len(conflicts), 1),
        )
        overall = round(
            term_score * 0.24
            + source_score * 0.12
            + confidence * 0.12
            + authority * 0.16
            + context_relevance * 0.16
            + citation_coverage * 0.14
            + conflict_awareness * 0.06,
            4,
        )
        return {
            "case_id": case.case_id,
            "category": case.category,
            "question": case.question,
            "expected_terms": case.expected_terms,
            "term_score": term_score,
            "source_score": source_score,
            "authority_score": authority,
            "context_relevance": context_relevance,
            "citation_coverage": citation_coverage,
            "conflict_awareness": conflict_awareness,
            "retrieval_confidence": confidence,
            "overall": overall,
            "retrieved_docs_count": answer.get("retrieved_docs_count", 0),
            "sources": sources,
            "conflicts": conflicts,
            "split": case.split,
            "failure_modes": self._failure_modes(
                term_score,
                source_score,
                authority,
                context_relevance,
                citation_coverage,
                conflict_awareness,
            ),
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
            expected_sources=[str(source) for source in item.get("expected_sources", [])],
            split=str(item.get("split") or "validation"),
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

    def _context_relevance(self, sources: List[Dict[str, Any]]) -> float:
        scores = [float(item.get("score") or 0) for item in sources]
        return round(sum(scores) / max(len(scores), 1), 3) if scores else 0.0

    def _citation_coverage(self, sources: List[Dict[str, Any]], expected_sources: List[str]) -> float:
        if not sources:
            return 0.0
        structurally_valid = sum(
            1
            for source in sources
            if source.get("source") and source.get("chunk_id") and source.get("content")
        ) / len(sources)
        if not expected_sources:
            return round(structurally_valid, 3)
        actual = " ".join(str(source.get("source") or "").lower() for source in sources)
        matched = sum(1 for expected in expected_sources if expected.lower() in actual)
        return round(structurally_valid * 0.5 + matched / max(len(expected_sources), 1) * 0.5, 3)

    def _failure_modes(
        self,
        term_score: float,
        source_score: float,
        authority: float,
        context_relevance: float,
        citation_coverage: float,
        conflict_awareness: float,
    ) -> List[str]:
        failures = []
        if term_score < 0.5:
            failures.append("答案未覆盖关键审计术语或控制点。")
        if source_score < 0.5:
            failures.append("召回来源数量不足，建议补充制度、底稿或日志样本。")
        if authority < 0.6:
            failures.append("权威来源占比偏低，需优先使用标准、制度和正式底稿。")
        if context_relevance < 0.5:
            failures.append("检索上下文相关性偏低，需要改写查询、过滤元数据或优化重排。")
        if citation_coverage < 0.7:
            failures.append("引用缺少可定位 chunk、内容摘要或预期来源覆盖。")
        if conflict_awareness < 0.7:
            failures.append("系统检出冲突证据但未正确转人工复核。")
        return failures or ["未发现明显 RAG 失败模式。"]

    def _closed_loop_suggestions(self, results: List[Dict[str, Any]]) -> List[str]:
        suggestions = []
        if any(item["term_score"] < 0.6 for item in results):
            suggestions.append("按失败用例补充审计术语同义词、控制目标和证据类型。")
        if any(item["source_score"] < 0.6 for item in results):
            suggestions.append("将企业制度、抽样底稿、系统导出和访谈纪要纳入知识库。")
        if any(item["authority_score"] < 0.6 for item in results):
            suggestions.append("为知识片段增加 source_type、owner、effective_date 和 authority_level 元数据。")
        if any(item["context_relevance"] < 0.6 for item in results):
            suggestions.append("为失败查询增加元数据过滤与独立 rerank 诊断，避免只依赖关键词命中。")
        if any(item["citation_coverage"] < 0.7 for item in results):
            suggestions.append("保存页码、章节、chunk_id 和预期来源标注，单独评测引用正确性。")
        return suggestions or ["保持当前 RAG 表现，继续纳入真实审计 badcase 做回归测试。"]

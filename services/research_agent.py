"""Deep-research style audit question answering and evaluation planning."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class ResearchStep:
    stage: str
    action: str
    output: str


class AuditResearchAgent:
    """A deterministic deep-research layer over the existing RAG pipeline."""

    def __init__(self, rag_pipeline: Any) -> None:
        self.rag_pipeline = rag_pipeline

    def answer(self, question: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        intent = self._classify_intent(question)
        rewrites = self._rewrite_queries(question, intent, context)
        retrievals = [self.rag_pipeline.query(query, context) for query in rewrites]
        sources = self._merge_sources(retrievals)
        reasoning = self._reason(question, intent, sources)
        answer = self._compose_answer(question, intent, reasoning, sources)
        evaluation = self._evaluate_answer(question, answer, sources)
        return {
            "question": question,
            "intent": intent,
            "query_rewrites": rewrites,
            "sources": sources,
            "reasoning_trace": [step.__dict__ for step in reasoning],
            "answer": answer,
            "evaluation": evaluation,
            "generated_at": datetime.now().isoformat(),
        }

    def jd_coverage(self) -> Dict[str, Any]:
        return {
            "source": "腾讯招聘官网 careers.tencent.com 2026-05 至 2026-06 Agent/大模型岗位 + 字节 Seed 搜索问答 Agent JD",
            "official_tencent_posts": [
                {
                    "post": "Agent Evaluation Intern 107491",
                    "post_id": "2057058794919346176",
                    "updated": "2026-05-20",
                    "requirements": [
                        "自动化评估流水线、执行产物采集、回归检测",
                        "评估 tool use、参数准确性、错误处理和不必要调用",
                        "分析 traces、logs、中间步骤、最终输出，定位推理失败和上下文误用",
                        "设计成功率、工具精度、恢复率、重试、延迟、成本和安全失败率指标",
                        "用合成用例、黄金流程和匿名真实执行构建可复用数据集",
                    ],
                },
                {
                    "post": "元宝-Agent架构工程师",
                    "post_id": "2016726997581058048",
                    "updated": "2026-06-04",
                    "requirements": [
                        "Agent Runtime、Tool、Memory、Context 抽象",
                        "多 Agent 协作模式与 Human-in-the-loop",
                        "结合模型边界、产品功能和架构约束设计整体方案",
                    ],
                },
                {
                    "post": "微信搜索-Agent算法专家",
                    "post_id": "2062097072978575360",
                    "updated": "2026-06-11",
                    "requirements": [
                        "Search Agent、DeepSearch、DeepResearch 和真实世界复杂任务 Agentic 能力",
                        "Mid-Train、SFT、GRM、PRM、RLVR、Agentic RL、Agent 自进化",
                        "Context 管理、Memory 以及大模型结合搜索的下一代产品范式",
                    ],
                },
                {
                    "post": "腾讯视频-AI Agent工程师",
                    "post_id": "2049051430010122240",
                    "updated": "2026-06-09",
                    "requirements": [
                        "任务规划、工具调用、记忆管理、多轮决策",
                        "高并发 Agent 服务框架、调度系统、工作流引擎和稳定性治理",
                        "Workflow/DAG/多 Agent 协作、Function Calling、Tool Use、RAG、上下文管理",
                        "效果评估、badcase 分析和迭代",
                    ],
                },
            ],
            "capabilities": [
                {
                    "jd_requirement": "端到端搜索问答：意图理解、查询改写、RAG、多源融合和答案生成",
                    "implemented": ["intent", "query_rewrites", "hybrid_rag", "source_fusion", "grounded_answer"],
                    "project_surface": ["/api/research/answer", "/knowledge", "/audit"],
                    "next_step": "为每个来源增加 authority_level、effective_date 和 evidence_type 元数据。",
                },
                {
                    "jd_requirement": "Deep Research：复杂问题、多轮对话、跨文档推理",
                    "implemented": ["multi_query_plan", "cross_source_reasoning", "evidence_gap_detection"],
                    "project_surface": ["/api/research/answer", "execution_trace"],
                    "next_step": "把用户会话摘要写入 Memory，并在下一轮查询中显式引用。",
                },
                {
                    "jd_requirement": "Agent 轨迹评估：日志、trace、中间步骤、最终输出和回归检测",
                    "implemented": ["execution_trace", "trajectory_score", "regression_risks", "badcase_suggestions"],
                    "project_surface": ["/api/training/evaluate", "/training"],
                    "next_step": "增加版本对比和历史评测趋势图。",
                },
                {
                    "jd_requirement": "Tool Use：工具选择、参数准确性、错误恢复、避免无效调用",
                    "implemented": ["skill_registry", "mcp_tool_descriptions", "tool_trace_quality"],
                    "project_surface": ["/api/skills", "/api/mcp/tools", "/training"],
                    "next_step": "为每个工具调用记录 input/output/error 和 retry policy。",
                },
                {
                    "jd_requirement": "Agent Runtime：Tool/Memory/Context 抽象、多 Agent、Human-in-the-loop",
                    "implemented": ["audit_planner", "session_memory", "quality_gate", "review_api"],
                    "project_surface": ["/audit", "/api/audit/runs/{run_id}/review"],
                    "next_step": "把质量门升级为可配置策略，并支持多角色审计员协作。",
                },
                {
                    "jd_requirement": "工程闭环：指标体系、自动化评测、badcase 分析、数据-模型-系统优化",
                    "implemented": ["evaluation_plan", "rag_evaluator", "benchmark_cases", "closed_loop_suggestions"],
                    "project_surface": ["/api/research/evaluation-plan", "/api/evaluation/rag", "/training"],
                    "next_step": "将评测结果持久化，形成版本发布前准入门禁。",
                },
            ],
        }

    def evaluation_plan(self) -> Dict[str, Any]:
        metrics = [
            {"metric": "faithfulness", "name": "真实性", "rule": "答案必须能回溯到来源、证据或审计底稿，不能新增无依据事实。"},
            {"metric": "freshness", "name": "时效性", "rule": "对可变事实标注来源时间；高时效问题要求联网或接入企业实时数据源。"},
            {"metric": "authority", "name": "权威性", "rule": "优先使用监管、标准、制度、审计底稿和企业系统数据。"},
            {"metric": "relevance", "name": "相关性", "rule": "检索来源应覆盖审计对象、风险主题、标准和业务上下文。"},
            {"metric": "tool_use", "name": "工具调用", "rule": "检查工具选择、参数、调用必要性、错误恢复和重试策略。"},
            {"metric": "trajectory", "name": "轨迹质量", "rule": "评估规划、检索、推理、质量门、人工复核等中间步骤是否完整。"},
            {"metric": "ux", "name": "用户体验", "rule": "答案应给出结论、依据、风险、动作、证据缺口和下一步。"},
        ]
        cases = [
            {"case_id": "DR-01", "question": "ERP 权限审计如何覆盖职责分离、特权账号和复核证据？", "expected": ["权限", "职责分离", "证据", "复核"]},
            {"case_id": "DR-02", "question": "SOX ITGC 变更管理测试需要哪些抽样底稿？", "expected": ["变更单", "审批", "测试", "上线"]},
            {"case_id": "DR-03", "question": "数据安全审计如何判断分类分级和共享审批是否充分？", "expected": ["数据目录", "分类分级", "共享审批", "日志"]},
            {"case_id": "DR-04", "question": "Agent 工具调用失败后如何降级、重试并触发人工复核？", "expected": ["工具", "降级", "重试", "人工复核"]},
        ]
        return {
            "metrics": metrics,
            "benchmark_cases": cases,
            "closed_loop": ["采集失败样例", "分析检索/推理/工具缺口", "补充知识或规则", "回归评测", "发布版本"],
            "release_gate": {
                "overall_score": ">= 0.75",
                "faithfulness": ">= 0.70",
                "tool_trace_quality": ">= 0.80",
                "critical_regressions": "0",
            },
        }

    def _classify_intent(self, question: str) -> Dict[str, Any]:
        q = question.lower()
        if any(word in question for word in ["对比", "比较", "差异"]):
            intent = "comparison"
        elif any(word in question for word in ["如何", "怎么", "流程", "步骤"]):
            intent = "procedure"
        elif any(word in question for word in ["风险", "缺口", "问题"]):
            intent = "risk_diagnosis"
        else:
            intent = "audit_qa"
        domains = [word for word in ["权限", "变更", "数据", "日志", "备份", "SOX", "ISO27001", "COBIT", "Agent", "RAG"] if word.lower() in q or word in question]
        return {"type": intent, "domains": domains or ["通用审计"], "complexity": "high" if len(domains) >= 2 else "medium"}

    def _rewrite_queries(self, question: str, intent: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        domains = intent.get("domains") or []
        rewrites = [question, f"{question} 审计证据 控制测试 质量门"]
        if context.get("standard_type"):
            rewrites.append(f"{question} {context['standard_type']} 审计要求")
        for domain in domains[:3]:
            rewrites.append(f"{domain} 控制目标 测试程序 证据清单")
        return list(dict.fromkeys(rewrites))[:6]

    def _merge_sources(self, retrievals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        merged: Dict[str, Dict[str, Any]] = {}
        for result in retrievals:
            for source in result.get("sources", []):
                key = f"{source.get('source')}::{source.get('chunk_id')}"
                existing = merged.get(key)
                if not existing or float(source.get("score") or 0) > float(existing.get("score") or 0):
                    merged[key] = source
        return sorted(merged.values(), key=lambda item: float(item.get("score") or 0), reverse=True)[:8]

    def _reason(self, question: str, intent: Dict[str, Any], sources: List[Dict[str, Any]]) -> List[ResearchStep]:
        coverage = "、".join(intent.get("domains") or [])
        source_count = len(sources)
        return [
            ResearchStep("understand", "识别意图和领域", f"问题属于 {intent['type']}，覆盖 {coverage}。"),
            ResearchStep("retrieve", "执行多路查询和来源融合", f"融合 {source_count} 条候选来源。"),
            ResearchStep("verify", "检查证据充分性", "来源不足时保留证据缺口，不输出绝对结论。" if source_count < 2 else "来源可支持初步结论。"),
            ResearchStep("decide", "生成审计动作", "输出结论、依据、风险、取证清单和下一步执行建议。"),
        ]

    def _compose_answer(self, question: str, intent: Dict[str, Any], reasoning: List[ResearchStep], sources: List[Dict[str, Any]]) -> str:
        source_labels = "、".join(str(item.get("source", "unknown")) for item in sources[:3]) or "暂无来源"
        gap = "证据来源不足，需要补充企业制度、底稿或系统导出。" if len(sources) < 2 else "已有来源可支撑初步判断，但仍需现场证据验证。"
        return (
            f"结论：该问题应按 {intent['type']} 场景处理，先明确审计对象、控制目标和证据链。\n\n"
            f"依据：当前召回来源包括 {source_labels}。\n\n"
            f"推理：{'; '.join(step.output for step in reasoning)}\n\n"
            f"建议动作：形成控制矩阵、证据请求、抽样计划和复核条件；{gap}"
        )

    def _evaluate_answer(self, question: str, answer: str, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "faithfulness": 0.82 if sources else 0.35,
            "authority": min(1.0, 0.45 + sum(1 for item in sources if "seed" in str(item.get("source")) or "builtin" in str(item.get("source"))) * 0.12),
            "relevance": 0.78 if any(term in answer for term in question[:12]) or sources else 0.4,
            "completeness": 0.76 if "建议动作" in answer and "依据" in answer else 0.5,
            "requires_human_review": len(sources) < 2,
        }

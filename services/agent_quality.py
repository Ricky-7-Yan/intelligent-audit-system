"""Interview-driven Agent quality diagnostics.

This service turns the concerns repeatedly seen in Agent/RAG interview
experiences into executable product diagnostics.  It intentionally reads real
runtime state instead of returning static resume text, so the project can
demonstrate the design decisions during an interview.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from services.agent_runtime import AgentRuntime
from services.conversation_memory import ConversationMemory
from services.evaluation_repository import EvaluationRunRepository
from services.skill_registry import SkillRegistry


@dataclass(frozen=True)
class QualityDimension:
    dimension_id: str
    name: str
    interview_signal: str
    design_answer: str


DIMENSIONS = [
    QualityDimension(
        "agent_runtime",
        "Agent 架构与长任务执行",
        "面试官会追问 Agent 是否只是 Chatbot、有没有规划/执行/反思/失败恢复。",
        "采用轻量 Plan/Execute/Reflect Runtime，而不是把业务逻辑藏在框架里；生产可迁移 LangGraph/Temporal。",
    ),
    QualityDimension(
        "rag_grounding",
        "RAG 召回、错召/漏召与幻觉控制",
        "面试官会追问切块、混合检索、重排、RAG 评测和证据不足时怎么办。",
        "审计场景需要标准编号和控制 ID 精确匹配，因此采用混合检索、来源置信度、质量门和补证任务。",
    ),
    QualityDimension(
        "tool_mcp",
        "Tool Use / MCP / Skill 治理",
        "面试官会追问工具 Schema、权限、缓存、熔断、MCP 与 Function Calling 区别。",
        "SkillRegistry 把工具升级为治理单元，包含 Schema、权限、TTL、熔断、日志和 MCP-style 描述。",
    ),
    QualityDimension(
        "evaluation_harness",
        "评测、发布门禁与自进化 Harness",
        "面试官会追问效果如何证明、怎么防止改坏、badcase 如何沉淀。",
        "Agent/RAG/Research 评测会持久化并和历史基线比较，Harness 只生成可验证提案，不自动上线。",
    ),
    QualityDimension(
        "memory_context",
        "Memory 与上下文压缩",
        "面试官会追问短期/长期记忆、上下文压缩、记忆污染和删除策略。",
        "记忆分为 Working、Episodic、Profile、Related，分别处理顺序、摘要、画像和相关历史召回。",
    ),
    QualityDimension(
        "production_engineering",
        "生产化工程质量",
        "面试官会追问是不是 demo、如何迁移数据库、异步任务、压测、日志和降级。",
        "当前保持本地可运行和透明持久化，生产可替换为数据库、对象存储、任务队列、Redis 与日志检索。",
    ),
]


class AgentQualityDiagnostics:
    """Build a single, executable quality report for interview-critical areas."""

    def __init__(
        self,
        evaluation_repository: EvaluationRunRepository,
        agent_runtime: AgentRuntime,
        skill_registry: SkillRegistry,
        conversation_memory: ConversationMemory,
    ) -> None:
        self.evaluation_repository = evaluation_repository
        self.agent_runtime = agent_runtime
        self.skill_registry = skill_registry
        self.conversation_memory = conversation_memory

    def report(self, rag_stats: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        observability = self.agent_runtime.observability()
        skill_metrics = self.skill_registry.metrics()
        memory_stats = self.conversation_memory.stats()
        eval_runs = self.evaluation_repository.list_runs(limit=50)
        rag_stats = rag_stats or {"total_documents": 0}

        dimensions = [
            self._agent_runtime_dimension(observability),
            self._rag_dimension(eval_runs, rag_stats),
            self._tool_dimension(skill_metrics),
            self._evaluation_dimension(eval_runs),
            self._memory_dimension(memory_stats),
            self._production_dimension(observability, skill_metrics, eval_runs, rag_stats),
        ]
        overall_score = round(sum(item["score"] for item in dimensions) / max(len(dimensions), 1), 1)

        return {
            "overall_score": overall_score,
            "readiness_label": self._label(overall_score),
            "dimensions": dimensions,
            "badcase_diagnostics": self._badcases(eval_runs, observability, skill_metrics),
            "tool_use_diagnostics": self._tool_use(skill_metrics),
            "rag_diagnostics": self._rag_diagnostics(eval_runs, rag_stats),
            "production_readiness": self._production_readiness(observability, skill_metrics, memory_stats, rag_stats),
            "interview_pitch": self._pitch(dimensions),
        }

    def _agent_runtime_dimension(self, observability: Dict[str, Any]) -> Dict[str, Any]:
        task_count = int(observability.get("tasks") or 0)
        tool_calls = int(observability.get("tool_calls") or 0)
        reflections = int(observability.get("reflections") or 0)
        score = 45 + min(task_count, 10) * 3 + min(tool_calls, 20) * 1.2 + min(reflections, 20) * 1.1
        gaps = []
        if task_count == 0:
            gaps.append("缺少可回放的 AgentRuntime 任务，面试时难证明不是 Chatbot。")
        if reflections == 0:
            gaps.append("缺少反思记录，建议运行任务或制造失败样例验证恢复链路。")
        return self._dimension(
            "agent_runtime",
            score,
            [
                f"任务数 {task_count}",
                f"工具调用 {tool_calls}",
                f"反思记录 {reflections}",
                f"P95 延迟 {observability.get('p95_latency_ms', 0)}ms",
            ],
            gaps,
            ["演示 /skills 里的任务计划、步骤、工具调用和反思。", "准备说明生产可迁移 LangGraph/Temporal。"],
        )

    def _rag_dimension(self, eval_runs: List[Dict[str, Any]], rag_stats: Dict[str, Any]) -> Dict[str, Any]:
        doc_count = int(rag_stats.get("total_documents") or rag_stats.get("total_chunks") or 0)
        latest_rag = next((run for run in eval_runs if run.get("run_type") == "rag"), None)
        score = 40 + min(doc_count, 20) * 1.5
        evidence = [f"RAG 文档/切片 {doc_count}"]
        gaps = []
        if latest_rag:
            metrics = latest_rag.get("metrics", {})
            score += float(metrics.get("overall_score") or 0) * 35
            evidence.append(f"最近 RAG 评测 {latest_rag.get('run_id')}，得分 {metrics.get('overall_score', 0)}")
            gate = latest_rag.get("release_gate", {})
            evidence.append(f"发布门禁 {gate.get('label') or gate.get('status')}")
            if gate.get("blockers"):
                gaps.extend(gate.get("blockers", []))
        else:
            gaps.append("缺少最近 RAG 评测，面试时难量化错召/漏召处理效果。")
        if doc_count == 0:
            gaps.append("知识库为空，RAG 只能讲设计，不能现场证明召回。")
        return self._dimension(
            "rag_grounding",
            score,
            evidence,
            gaps,
            ["补充 golden set 并运行 /api/evaluation/rag。", "为高频审计标准补充控制编号和证据样本。"],
        )

    def _tool_dimension(self, skill_metrics: Dict[str, Any]) -> Dict[str, Any]:
        total_runs = int(skill_metrics.get("total_runs") or 0)
        success_rate = float(skill_metrics.get("success_rate") or 0)
        open_circuits = int(skill_metrics.get("open_circuits") or 0)
        score = 45 + success_rate * 35 + min(total_runs, 30) * 0.7 - open_circuits * 8
        gaps = []
        if total_runs == 0:
            gaps.append("缺少工具运行日志，建议现场触发 RAG/报告/控制映射等 Skill。")
        if success_rate < 0.9 and total_runs:
            gaps.append("工具成功率低于 90%，需要定位失败 Skill。")
        if open_circuits:
            gaps.append("存在打开的熔断器，说明工具可靠性需要恢复。")
        evidence = [
            f"Skill 运行 {total_runs}",
            f"成功率 {round(success_rate * 100)}%",
            f"缓存命中 {skill_metrics.get('cache_hits', 0)}",
            f"打开熔断 {open_circuits}",
        ]
        return self._dimension(
            "tool_mcp",
            score,
            evidence,
            gaps,
            ["演示 /api/mcp/tools 的 inputSchema 与权限声明。", "解释 TTL 缓存和熔断如何降低延迟与故障放大。"],
        )

    def _evaluation_dimension(self, eval_runs: List[Dict[str, Any]]) -> Dict[str, Any]:
        run_count = len(eval_runs)
        blocked = [run for run in eval_runs if (run.get("release_gate") or {}).get("status") == "blocked"]
        review = [run for run in eval_runs if (run.get("release_gate") or {}).get("status") == "review"]
        score = 42 + min(run_count, 20) * 2.2 - len(blocked) * 8 - len(review) * 2
        gaps = []
        if run_count == 0:
            gaps.append("缺少评测历史，面试时难证明效果和回归控制。")
        if blocked:
            gaps.append(f"存在 {len(blocked)} 个 blocked release gate，需要优先处理。")
        evidence = [
            f"评测记录 {run_count}",
            f"review 门禁 {len(review)}",
            f"blocked 门禁 {len(blocked)}",
        ]
        return self._dimension(
            "evaluation_harness",
            score,
            evidence,
            gaps,
            ["把最新 blocker 转成 badcase，再运行回归评测。", "说明 Harness 只生成提案，不自动上线。"],
        )

    def _memory_dimension(self, memory_stats: Dict[str, Any]) -> Dict[str, Any]:
        sessions = int(memory_stats.get("sessions") or 0)
        turns = int(memory_stats.get("turns") or 0)
        episodes = int(memory_stats.get("episodes") or 0)
        score = 45 + min(sessions, 10) * 2 + min(turns, 50) * 0.5 + min(episodes, 20) * 1.2
        gaps = []
        if sessions == 0:
            gaps.append("缺少会话记忆样本，面试时无法演示上下文保留。")
        if turns > 20 and episodes == 0:
            gaps.append("会话轮次较多但没有 episode，建议触发压缩验证上下文治理。")
        evidence = [f"会话 {sessions}", f"轮次 {turns}", f"工作消息 {memory_stats.get('working_messages', 0)}", f"episodes {episodes}"]
        return self._dimension(
            "memory_context",
            score,
            evidence,
            gaps,
            ["准备解释为什么 Working/Episodic/Profile/Related 不能混成一个向量库。", "补充记忆删除、脱敏和冲突检测的生产演进。"],
        )

    def _production_dimension(
        self,
        observability: Dict[str, Any],
        skill_metrics: Dict[str, Any],
        eval_runs: List[Dict[str, Any]],
        rag_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        checks = self._production_readiness(observability, skill_metrics, self.conversation_memory.stats(), rag_stats)
        passed = sum(1 for item in checks if item["status"] == "pass")
        score = 35 + passed / max(len(checks), 1) * 60
        gaps = [item["gap"] for item in checks if item["status"] != "pass"]
        evidence = [f"{passed}/{len(checks)} 项生产化检查通过", f"评测记录 {len(eval_runs)}", f"RAG 文档 {rag_stats.get('total_documents', 0)}"]
        return self._dimension(
            "production_engineering",
            score,
            evidence,
            gaps,
            ["按数据库、对象存储、队列、Redis、日志检索、真实 MCP Server 六层讲生产化演进。"],
        )

    def _dimension(
        self,
        dimension_id: str,
        score: float,
        evidence: List[str],
        gaps: List[str],
        next_actions: List[str],
    ) -> Dict[str, Any]:
        dimension = next(item for item in DIMENSIONS if item.dimension_id == dimension_id)
        normalized = round(max(0, min(100, score)), 1)
        return {
            "dimension_id": dimension.dimension_id,
            "name": dimension.name,
            "score": normalized,
            "status": self._status(normalized),
            "interview_signal": dimension.interview_signal,
            "design_answer": dimension.design_answer,
            "evidence": evidence,
            "gaps": gaps,
            "next_actions": next_actions,
        }

    def _badcases(
        self,
        eval_runs: List[Dict[str, Any]],
        observability: Dict[str, Any],
        skill_metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        badcases: List[Dict[str, Any]] = []
        for run in eval_runs[:20]:
            gate = run.get("release_gate") or {}
            if gate.get("blockers"):
                badcases.append(
                    {
                        "source": "release_gate",
                        "severity": gate.get("status", "review"),
                        "title": f"{run.get('run_type')} {run.get('run_id')} 门禁未通过",
                        "signals": gate.get("blockers", []),
                        "interview_answer": "这类问题会进入 badcase backlog，先修复 blocker，再跑同类型回归。",
                    }
                )
        for skill, count in (skill_metrics.get("failures_by_skill") or {}).items():
            badcases.append(
                {
                    "source": "tool_reliability",
                    "severity": "review",
                    "title": f"{skill} 工具失败 {count} 次",
                    "signals": ["检查 Schema、权限、超时、外部依赖和熔断状态。"],
                    "interview_answer": "工具失败不会静默吞掉，会记录 error_type 并进入 Runtime reflection。",
                }
            )
        if int(observability.get("blocked_tasks") or 0):
            badcases.append(
                {
                    "source": "safety_gate",
                    "severity": "blocked",
                    "title": f"{observability.get('blocked_tasks')} 个任务被安全门阻断",
                    "signals": ["说明高风险/敏感操作没有自动执行。"],
                    "interview_answer": "这体现 Human-in-the-loop 边界，高风险任务必须复核。",
                }
            )
        return badcases[:12]

    def _tool_use(self, skill_metrics: Dict[str, Any]) -> Dict[str, Any]:
        volumes = skill_metrics.get("volume_by_skill") or {}
        failures = skill_metrics.get("failures_by_skill") or {}
        ranked = sorted(volumes.items(), key=lambda item: item[1], reverse=True)[:8]
        return {
            "selection_contract": [
                "工具必须有 inputSchema。",
                "工具必须声明权限和超时。",
                "失败必须记录 error_type。",
                "高频工具需要 TTL 缓存。",
                "连续失败必须进入熔断。",
            ],
            "top_tools": [
                {
                    "skill": skill,
                    "runs": count,
                    "failures": failures.get(skill, 0),
                    "success_rate": round((count - failures.get(skill, 0)) / max(count, 1), 3),
                }
                for skill, count in ranked
            ],
            "cache_hits": skill_metrics.get("cache_hits", 0),
            "circuits": skill_metrics.get("circuits", {}),
        }

    def _rag_diagnostics(self, eval_runs: List[Dict[str, Any]], rag_stats: Dict[str, Any]) -> Dict[str, Any]:
        rag_runs = [run for run in eval_runs if run.get("run_type") == "rag"]
        latest = rag_runs[0] if rag_runs else None
        return {
            "documents": rag_stats.get("total_documents", 0),
            "chunks": rag_stats.get("total_chunks", rag_stats.get("total_documents", 0)),
            "latest_eval": latest,
            "debug_playbook": [
                "先确认 golden 文档是否存在于知识库。",
                "再确认切块是否保留标准条款、控制编号和证据字段。",
                "检查目标文档是否进入 top-k。",
                "若召回但答案错，检查重排和生成阶段。",
                "若证据不足，输出 missing evidence 而不是强结论。",
            ],
        }

    def _production_readiness(
        self,
        observability: Dict[str, Any],
        skill_metrics: Dict[str, Any],
        memory_stats: Dict[str, Any],
        rag_stats: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        checks = [
            ("runtime_trace", int(observability.get("tasks") or 0) > 0, "缺少可回放 Agent 任务。"),
            ("tool_logs", int(skill_metrics.get("total_runs") or 0) > 0, "缺少工具运行日志。"),
            ("tool_resilience", float(skill_metrics.get("success_rate") or 0) >= 0.8, "工具成功率低于 80%。"),
            ("rag_knowledge", int(rag_stats.get("total_documents") or 0) > 0, "RAG 知识库为空。"),
            ("memory_state", int(memory_stats.get("sessions") or 0) > 0, "缺少会话记忆样本。"),
            ("latency_visible", float(observability.get("avg_latency_ms") or 0) >= 0, "缺少延迟指标。"),
            ("safety_gate", "safety_review_rate" in observability, "缺少安全门指标。"),
            ("release_gate", bool(self.evaluation_repository.list_runs(limit=1)), "缺少评测和发布门禁记录。"),
        ]
        return [
            {"check": name, "status": "pass" if passed else "gap", "gap": "" if passed else gap}
            for name, passed, gap in checks
        ]

    def _pitch(self, dimensions: Iterable[Dict[str, Any]]) -> List[str]:
        weak = [item for item in dimensions if item["score"] < 70]
        return [
            "我把面经里高频追问的 Agent 架构、RAG、Tool Use、评测、Memory 和生产化都做成了可运行诊断，而不是只写在文档里。",
            "每个诊断项都绑定真实运行数据：任务、工具日志、评测门禁、RAG 文档和记忆状态。",
            "如果面试官追问缺点，我会直接展示 gap 和 next action，说明项目边界和生产化演进。",
            f"当前最需要补强的是：{weak[0]['name'] if weak else '继续积累真实 badcase 和生产压测'}。",
        ]

    def _status(self, score: float) -> str:
        if score >= 85:
            return "strong"
        if score >= 70:
            return "ready"
        if score >= 55:
            return "needs_evidence"
        return "gap"

    def _label(self, score: float) -> str:
        if score >= 85:
            return "面试强证据"
        if score >= 70:
            return "可演示可解释"
        if score >= 55:
            return "需要补证"
        return "证据不足"


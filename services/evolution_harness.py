"""Self-evolution and JD-alignment harness for the audit agent platform."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List


JD_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "company": "ByteDance Seed / Agent Systems",
        "focus": "Agent harness、执行环境、沙箱、工具集成、长周期任务评测",
        "keywords": ["harness", "sandbox", "tool integration", "benchmark", "long-horizon"],
        "project_evidence": ["AgentRuntime", "SkillRegistry", "SafetyGate", "EvaluationRunRepository"],
        "implemented": True,
        "gap": "当前为本地轻量执行环境；生产级沙箱可继续扩展 Docker/Kubernetes worker。",
    },
    {
        "company": "ByteDance Seed / 大模型人才校招",
        "focus": "Code Agent RL、Test-time Scaling、Long Horizon Task、Memory、Search Agent",
        "keywords": ["test-time scaling", "memory", "search agent", "multi-agent rl"],
        "project_evidence": ["ConversationMemory", "HybridIntentRouter", "Agentic RAG", "runtime reflections"],
        "implemented": True,
        "gap": "已实现记忆/路由/反思；RL 训练与真实大规模实验需要算力和业务数据后续接入。",
    },
    {
        "company": "Tencent / 青云计划与企业级 Agent",
        "focus": "智能体、强化学习、Agent 前沿评测、开放域任务协同进化、经验单元沉淀",
        "keywords": ["multi-agent", "experience unit", "evaluation", "memory"],
        "project_evidence": ["multi-agent routing", "episodic memory", "baseline regression gate"],
        "implemented": True,
        "gap": "已沉淀审计域经验单元；开放域泛化可增加跨场景 benchmark。",
    },
    {
        "company": "Tencent / AI Agent 测试与评测",
        "focus": "任务完成率、多轮对话质量、工具调用准确性、自动化与人工评测、失败归因",
        "keywords": ["task success", "tool accuracy", "failure analysis", "human evaluation"],
        "project_evidence": ["evaluation runs", "release gate", "reflection issues", "human review workflow"],
        "implemented": True,
        "gap": "已具备量化与人工复核；可继续接入 AgentBench/TAU-bench 风格公开样例。",
    },
    {
        "company": "Alibaba / AI Agent 算法工程师",
        "focus": "Agent 全生命周期、SFT/RL、Planning、多步推理、RAG、工具调用、端到端评测",
        "keywords": ["lifecycle", "planning", "RAG", "tool calling", "post-training"],
        "project_evidence": ["audit lifecycle", "planner/control/evidence/remediation agents", "RAG evaluator"],
        "implemented": True,
        "gap": "已覆盖应用工程主链路；SFT/RL 属于训练侧增强，需要真实标注集和 GPU 训练计划。",
    },
    {
        "company": "Alibaba / AI Agent 优化工程师",
        "focus": "Prompt 工程化、Agent 编排、任务规划、Function Calling/MCP、业务落地",
        "keywords": ["prompt engineering", "orchestration", "Function Calling", "MCP"],
        "project_evidence": ["MCP-style tools", "Skill schema", "FastAPI endpoints", "audit delivery package"],
        "implemented": True,
        "gap": "本项目以 MCP 风格工具描述实现；若对接真实 MCP server，可复用现有 SkillRegistry。",
    },
]


class EvolutionHarness:
    """Derives self-improvement proposals from runtime, eval, memory, and JD signals."""

    def __init__(self, evaluation_repository, agent_runtime, skill_registry, conversation_memory) -> None:
        self.evaluation_repository = evaluation_repository
        self.agent_runtime = agent_runtime
        self.skill_registry = skill_registry
        self.conversation_memory = conversation_memory

    def report(self) -> Dict[str, Any]:
        eval_runs = self.evaluation_repository.list_runs(limit=30)
        observability = self.agent_runtime.observability()
        skill_metrics = self.skill_registry.metrics()
        memory_stats = self.conversation_memory.stats()
        coverage = self.jd_coverage()
        risks = self._regression_risks(eval_runs, observability, skill_metrics)
        proposals = self._proposals(eval_runs, observability, skill_metrics, memory_stats, risks)
        loops = self._harness_loops(proposals)
        return {
            "generated_at": datetime.now().isoformat(),
            "maturity_score": self._maturity_score(coverage, risks, observability, memory_stats),
            "jd_coverage": coverage,
            "runtime_signals": {
                "tasks": observability.get("tasks", 0),
                "tool_success_rate": observability.get("tool_success_rate", 0),
                "avg_latency_ms": observability.get("avg_latency_ms", 0),
                "reflections": observability.get("reflections", 0),
                "retry_count": observability.get("retry_count", 0),
                "memory_sessions": memory_stats.get("sessions", 0),
                "episodes": memory_stats.get("episodes", 0),
                "skills": skill_metrics.get("skills", 0),
                "cache_hits": skill_metrics.get("cache_hits", 0),
            },
            "regression_risks": risks,
            "evolution_proposals": proposals,
            "harness_loops": loops,
        }

    def jd_coverage(self) -> Dict[str, Any]:
        implemented = [item for item in JD_REQUIREMENTS if item["implemented"]]
        return {
            "items": JD_REQUIREMENTS,
            "covered": len(implemented),
            "total": len(JD_REQUIREMENTS),
            "coverage_rate": round(len(implemented) / max(len(JD_REQUIREMENTS), 1), 3),
            "remaining_gaps": [item["gap"] for item in JD_REQUIREMENTS if item.get("gap")],
        }

    def _regression_risks(
        self,
        eval_runs: List[Dict[str, Any]],
        observability: Dict[str, Any],
        skill_metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        risks: List[Dict[str, Any]] = []
        for run in eval_runs[:10]:
            gate = run.get("release_gate") or {}
            if gate.get("status") in {"review", "blocked"}:
                risks.append(
                    {
                        "type": "evaluation_gate",
                        "severity": "high" if gate.get("status") == "blocked" else "medium",
                        "signal": run.get("run_id"),
                        "reason": "；".join(gate.get("blockers") or ["评测结果需要复核"]),
                    }
                )
        if float(observability.get("tool_success_rate") or 1) < 0.85 and observability.get("tool_calls"):
            risks.append(
                {
                    "type": "tool_reliability",
                    "severity": "medium",
                    "signal": f"tool_success_rate={observability.get('tool_success_rate')}",
                    "reason": "工具成功率低于 85%，建议进入失败归因与重试策略调参。",
                }
            )
        if int(skill_metrics.get("open_circuits") or 0) > 0:
            risks.append(
                {
                    "type": "skill_circuit",
                    "severity": "high",
                    "signal": f"open_circuits={skill_metrics.get('open_circuits')}",
                    "reason": "存在熔断 Skill，需要检查依赖、超时或输入 schema。",
                }
            )
        return risks[:8]

    def _proposals(
        self,
        eval_runs: List[Dict[str, Any]],
        observability: Dict[str, Any],
        skill_metrics: Dict[str, Any],
        memory_stats: Dict[str, Any],
        risks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        proposals = [
            {
                "proposal_id": "HNS-01",
                "title": "把失败反思转成可回归的审计任务样例",
                "trigger": f"{observability.get('reflections', 0)} 条 runtime reflection",
                "action": "从 needs_review/blocked 任务中抽取 objective、上下文、期望证据，写入下一轮 agent/rag evaluation cases。",
                "validation": "新增样例后执行 /api/training/evaluate 与 /api/evaluation/rag，release_gate 必须 pass 或 review 无新增 blocker。",
                "impact": "提升长周期审计任务的稳定性和可解释性。",
            },
            {
                "proposal_id": "HNS-02",
                "title": "建立 Tool Calling 准确率与熔断恢复 playbook",
                "trigger": f"tool_success_rate={observability.get('tool_success_rate', 0)}，open_circuits={skill_metrics.get('open_circuits', 0)}",
                "action": "对失败 Skill 汇总 error_type、input_schema、duration_ms，生成最小复现实例与恢复建议。",
                "validation": "同输入重复运行时成功率提升，且 cache/circuit 指标不退化。",
                "impact": "对齐腾讯/阿里 JD 中的工具调用准确性、失败定位和工程治理要求。",
            },
            {
                "proposal_id": "HNS-03",
                "title": "沉淀审计经验单元，驱动跨会话自进化",
                "trigger": f"sessions={memory_stats.get('sessions', 0)}，episodes={memory_stats.get('episodes', 0)}",
                "action": "把高质量整改建议、证据缺口、控制测试模板沉淀为 profile/episode memory，并在相似审计场景自动召回。",
                "validation": "相同审计域二次提问时，答案包含历史标准、系统、风险主题且不引入冲突事实。",
                "impact": "对齐青云计划的经验单元沉淀、Memory 驱动个性化与复杂 Agent 泛化研究。",
            },
        ]
        if risks:
            proposals.insert(
                0,
                {
                    "proposal_id": "HNS-00",
                    "title": "优先处理当前回归风险",
                    "trigger": f"{len(risks)} 个退化/阻断信号",
                    "action": "先关闭 release_gate blocker，再允许新能力合入，避免 harness 更新带来负收益。",
                    "validation": "所有 high 风险转为 closed，最近一次评测无 baseline regression。",
                    "impact": "符合 Self-Harness 的弱点挖掘、候选修改、回归验证闭环。",
                },
            )
        if not eval_runs:
            proposals.append(
                {
                    "proposal_id": "HNS-04",
                    "title": "创建首个 Agent Benchmark 基线",
                    "trigger": "尚无持久化评测记录",
                    "action": "运行默认 agent benchmark，保存 baseline，后续所有优化都与该 baseline 比较。",
                    "validation": "data/evaluation_runs 生成 baseline_created 记录。",
                    "impact": "避免只有 demo，没有可量化演进证据。",
                }
            )
        return proposals

    def _harness_loops(self, proposals: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "loop": "Weakness Mining",
                "description": "从评测 release gate、runtime reflection、Skill run log 中挖掘失败模式。",
                "artifacts": ["data/evaluation_runs", "data/agent_runtime", "data/skill_runs.jsonl"],
            },
            {
                "loop": "Harness Proposal",
                "description": "把失败模式转成最小可验证的提示、工具 schema、记忆、路由或测试集改动建议。",
                "artifacts": [proposal["proposal_id"] for proposal in proposals],
            },
            {
                "loop": "Proposal Validation",
                "description": "通过 baseline regression、质量门、人工复核和 UI smoke test 验证改动没有负收益。",
                "artifacts": ["/api/training/evaluate", "/api/evaluation/rag", "/api/agent/observability"],
            },
        ]

    def _maturity_score(
        self,
        coverage: Dict[str, Any],
        risks: List[Dict[str, Any]],
        observability: Dict[str, Any],
        memory_stats: Dict[str, Any],
    ) -> int:
        score = 58
        score += int(coverage.get("coverage_rate", 0) * 22)
        if observability.get("tool_calls"):
            score += min(8, int(float(observability.get("tool_success_rate") or 0) * 8))
        if memory_stats.get("sessions"):
            score += 6
        score -= sum(8 if item["severity"] == "high" else 4 for item in risks)
        return max(0, min(100, score))

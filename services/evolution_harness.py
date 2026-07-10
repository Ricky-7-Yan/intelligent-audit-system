"""Self-evolution, AgentOps, and JD-alignment harness for AuditPilot.

The harness turns runtime traces, eval runs, tool health, memory signals, and
market-facing Agent engineering requirements into verifiable improvement work.
It is intentionally deterministic: the output can be used by UI, docs, and
interview demos without relying on a model call.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


MARKET_RADAR: List[Dict[str, Any]] = [
    {
        "trend_id": "MR-01",
        "name": "AgentOps：生产轨迹、评测与回归闭环",
        "why_it_matters": "企业不再只看 demo，而是看 trace、latency、cost、failure mode、human review 和持续回归。",
        "signals": ["observability", "evals", "release gate", "production traces", "failure analysis"],
        "project_mapping": ["AgentRuntime.observability", "EvaluationRunRepository", "release_gate", "reflection"],
        "current_level": "implemented",
        "next_gap": "把失败轨迹自动沉淀为 benchmark backlog，并支持一键生成修复任务。",
    },
    {
        "trend_id": "MR-02",
        "name": "Harness / Sandbox：可控执行环境与长周期任务验证",
        "why_it_matters": "字节、腾讯、阿里相关 Agent 岗位都强调任务执行环境、工具治理、长链路可靠性和自动评测。",
        "signals": ["harness", "sandbox", "long-horizon task", "tool governance", "guardrails"],
        "project_mapping": ["SafetyGate", "SkillRegistry", "AgentTask envelope", "tool circuit breaker"],
        "current_level": "implemented",
        "next_gap": "增加真实隔离 worker、权限分级和跨任务预算控制。",
    },
    {
        "trend_id": "MR-03",
        "name": "MCP / Tool Use：工具契约、权限与高效上下文装载",
        "why_it_matters": "市场正在从简单 function calling 走向标准化工具协议、工具注册表、权限注解和按需工具装载。",
        "signals": ["MCP", "tool schema", "permissions", "tool routing", "context efficiency"],
        "project_mapping": ["mcp_tools()", "Skill schema", "tool annotations", "SafetyGate"],
        "current_level": "implemented",
        "next_gap": "接入真实 MCP server，并记录 tool selection precision。",
    },
    {
        "trend_id": "MR-04",
        "name": "多 Agent 编排：角色分工、协作计划与人工复核",
        "why_it_matters": "复杂业务 Agent 更强调 planner / researcher / tool / verifier / human-in-loop 的分工和责任边界。",
        "signals": ["multi-agent", "planner", "verifier", "human-in-the-loop", "collaboration plan"],
        "project_mapping": ["HybridIntentRouter", "collaboration_plan", "review workflow"],
        "current_level": "implemented",
        "next_gap": "把协作角色的输出拆成可单独评测的 role-level trace。",
    },
    {
        "trend_id": "MR-05",
        "name": "选择性记忆与经验单元：从上下文窗口走向可治理记忆",
        "why_it_matters": "真实 Agent 要能沉淀经验、召回相似任务，同时避免污染、过期和隐私泄漏。",
        "signals": ["memory", "episodic memory", "experience unit", "retrieval", "profile"],
        "project_mapping": ["ConversationMemory", "memory stats", "session profile"],
        "current_level": "partial",
        "next_gap": "增加经验单元评分、过期策略、冲突检测和隐私标签。",
    },
    {
        "trend_id": "MR-06",
        "name": "行业工作台：垂直领域 Agent 从聊天走向可交付工作流",
        "why_it_matters": "成熟产品更像行业工作台：有数据、有审批、有交付包、有审计轨迹，而不是纯聊天窗口。",
        "signals": ["domain workbench", "workflow", "deliverables", "auditability", "governance"],
        "project_mapping": ["audit workspace", "delivery package", "control library", "evidence workflow"],
        "current_level": "implemented",
        "next_gap": "加强跨项目组合视图和真实企业系统连接器。",
    },
]


JD_REQUIREMENTS: List[Dict[str, Any]] = [
    {
        "company": "ByteDance / Seed Agent Systems",
        "focus": "Agent harness、执行环境、沙箱、工具集成、长周期任务评测",
        "keywords": ["harness", "sandbox", "tool integration", "benchmark", "long-horizon"],
        "project_evidence": ["AgentRuntime", "SkillRegistry", "SafetyGate", "EvaluationRunRepository"],
        "implemented": True,
        "gap": "当前为本地轻量执行环境；生产级沙箱可继续扩展 Docker/Kubernetes worker。",
    },
    {
        "company": "ByteDance / Code Agent & Search Agent",
        "focus": "Test-time scaling、Memory、Search Agent、多步推理、长链路任务",
        "keywords": ["test-time scaling", "memory", "search agent", "multi-step reasoning"],
        "project_evidence": ["ConversationMemory", "HybridIntentRouter", "Agentic RAG", "runtime reflection"],
        "implemented": True,
        "gap": "已具备记忆/路由/反思；RL 或大规模训练需真实标注集与算力接入。",
    },
    {
        "company": "Tencent / 青云计划与企业级 Agent",
        "focus": "多 Agent 协作、强化学习、前沿评测、开放域任务协同进化、经验单元沉淀",
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
        "gap": "可继续接入 AgentBench / TAU-bench 风格公开样例。",
    },
    {
        "company": "Alibaba / AI Agent 算法工程",
        "focus": "Agent 全生命周期、SFT/RL、Planning、多步推理、RAG、工具调用、端到端评测",
        "keywords": ["lifecycle", "planning", "RAG", "tool calling", "post-training"],
        "project_evidence": ["audit lifecycle", "planner/control/evidence/remediation agents", "RAG evaluator"],
        "implemented": True,
        "gap": "主链路已覆盖；SFT/RL 属训练侧增强，需要真实标注集和 GPU 训练计划。",
    },
    {
        "company": "Alibaba / AI Agent 优化工程",
        "focus": "Prompt 工程化、Agent 编排、任务规划、Function Calling/MCP、业务落地",
        "keywords": ["prompt engineering", "orchestration", "Function Calling", "MCP"],
        "project_evidence": ["MCP-style tools", "Skill schema", "FastAPI endpoints", "audit delivery package"],
        "implemented": True,
        "gap": "若对接真实 MCP server，可复用当前 SkillRegistry 与 SafetyGate。",
    },
]


class EvolutionHarness:
    """Derives self-improvement proposals from runtime, eval, memory, and market signals."""

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
        market = self.market_alignment(observability, skill_metrics, memory_stats)
        risks = self._regression_risks(eval_runs, observability, skill_metrics)
        backlog = self.benchmark_backlog(eval_runs, observability, skill_metrics, risks)
        proposals = self._proposals(eval_runs, observability, skill_metrics, memory_stats, risks, backlog)
        loops = self._harness_loops(proposals)
        return {
            "generated_at": datetime.now().isoformat(),
            "maturity_score": self._maturity_score(coverage, risks, observability, memory_stats, market),
            "market_radar": market,
            "jd_coverage": coverage,
            "runtime_signals": {
                "tasks": observability.get("tasks", 0),
                "active_tasks": observability.get("active_tasks", 0),
                "blocked_tasks": observability.get("blocked_tasks", 0),
                "tool_calls": observability.get("tool_calls", 0),
                "tool_success_rate": observability.get("tool_success_rate", 0),
                "avg_latency_ms": observability.get("avg_latency_ms", 0),
                "p95_latency_ms": observability.get("p95_latency_ms", 0),
                "reflections": observability.get("reflections", 0),
                "retry_count": observability.get("retry_count", 0),
                "memory_sessions": memory_stats.get("sessions", 0),
                "working_messages": memory_stats.get("working_messages", 0),
                "episodes": memory_stats.get("episodes", 0),
                "skills": skill_metrics.get("skills", 0),
                "cache_hits": skill_metrics.get("cache_hits", 0),
                "open_circuits": skill_metrics.get("open_circuits", 0),
            },
            "trajectory_protocol": self.trajectory_protocol(observability),
            "regression_risks": risks,
            "benchmark_backlog": backlog,
            "evolution_proposals": proposals,
            "evolution_control_plane": self.control_plane(proposals, risks),
            "harness_loops": loops,
        }

    def market_alignment(
        self,
        observability: Dict[str, Any],
        skill_metrics: Dict[str, Any],
        memory_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        items = []
        for trend in MARKET_RADAR:
            level = trend["current_level"]
            if trend["trend_id"] == "MR-01" and not observability.get("tool_calls"):
                level = "partial"
            if trend["trend_id"] == "MR-05" and not memory_stats.get("episodes"):
                level = "partial"
            if trend["trend_id"] == "MR-03" and int(skill_metrics.get("skills") or 0) < 8:
                level = "partial"
            items.append({**trend, "current_level": level})
        implemented = sum(1 for item in items if item["current_level"] == "implemented")
        return {
            "items": items,
            "implemented": implemented,
            "total": len(items),
            "alignment_rate": round(implemented / max(len(items), 1), 3),
            "top_gaps": [item["next_gap"] for item in items if item["current_level"] != "implemented"][:5],
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

    def trajectory_protocol(self, observability: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "protocol": "audit-agent-trajectory-v2",
            "purpose": "把多 Agent 任务轨迹标准化为可观测、可评测、可回放的数据结构。",
            "required_fields": [
                "task_id",
                "agent_role",
                "step_id",
                "skill",
                "input_hash",
                "status",
                "duration_ms",
                "attempts",
                "safety_gate",
                "reflection",
                "artifact_refs",
            ],
            "observed": {
                "tasks": observability.get("tasks", 0),
                "tool_calls": observability.get("tool_calls", 0),
                "reflections": observability.get("reflections", 0),
                "retry_count": observability.get("retry_count", 0),
            },
            "quality_checks": [
                "每个 tool_call 必须能追溯到 step_id 与 skill。",
                "失败或重试必须生成 reflection。",
                "产物必须保留 source_run_id。",
                "进入发布门禁前必须完成 baseline comparison。",
            ],
        }

    def benchmark_backlog(
        self,
        eval_runs: List[Dict[str, Any]],
        observability: Dict[str, Any],
        skill_metrics: Dict[str, Any],
        risks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        backlog: List[Dict[str, Any]] = []
        for risk in risks[:5]:
            backlog.append(
                {
                    "case_id": f"BB-{len(backlog) + 1:03d}",
                    "source": risk["type"],
                    "priority": risk["severity"],
                    "question": f"复现并修复 {risk['type']}：{risk['reason']}",
                    "expected_terms": ["根因", "复现", "修复", "回归验证"],
                    "target_eval": "agent",
                    "acceptance": "新增用例通过，且 release_gate 不新增 blocker。",
                }
            )
        if not eval_runs:
            backlog.append(
                {
                    "case_id": "BB-BASELINE-001",
                    "source": "missing_baseline",
                    "priority": "high",
                    "question": "建立 Agent 基线：ERP 权限审计范围、证据、控制测试、整改闭环。",
                    "expected_terms": ["范围", "证据", "控制测试", "整改", "复核"],
                    "target_eval": "agent",
                    "acceptance": "生成 baseline_created 记录。",
                }
            )
        if float(observability.get("tool_success_rate") or 1) < 0.9 and observability.get("tool_calls"):
            backlog.append(
                {
                    "case_id": "BB-TOOL-001",
                    "source": "tool_reliability",
                    "priority": "medium",
                    "question": "验证工具失败重试、熔断恢复和错误归因是否完整记录。",
                    "expected_terms": ["重试", "熔断", "错误类型", "恢复动作"],
                    "target_eval": "runtime",
                    "acceptance": "tool_success_rate 提升且 open_circuits 不增加。",
                }
            )
        if int(skill_metrics.get("cache_hits") or 0) == 0:
            backlog.append(
                {
                    "case_id": "BB-CACHE-001",
                    "source": "tool_efficiency",
                    "priority": "low",
                    "question": "验证重复工具调用是否命中缓存，减少 token 与耗时浪费。",
                    "expected_terms": ["缓存", "重复调用", "耗时", "命中率"],
                    "target_eval": "runtime",
                    "acceptance": "相同输入二次执行出现 cache_hit。",
                }
            )
        return backlog[:8]

    def control_plane(self, proposals: List[Dict[str, Any]], risks: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {
            "mode": "risk_first" if risks else "continuous_improvement",
            "release_policy": {
                "block_on_high_risk": True,
                "require_eval_for_prompt_or_tool_change": True,
                "require_human_review_for_safety_or_evidence_gap": True,
            },
            "lanes": [
                {"lane": "Observe", "owner": "AgentOps", "input": "trace / eval / memory / skill logs"},
                {"lane": "Mine Weakness", "owner": "Harness", "input": "regression_risks + benchmark_backlog"},
                {"lane": "Propose", "owner": "Evolution", "input": "evolution_proposals"},
                {"lane": "Validate", "owner": "Release Gate", "input": "agent eval + rag eval + UI smoke"},
                {"lane": "Ship or Rollback", "owner": "Human Reviewer", "input": "release_gate + evidence"},
            ],
            "actionable_proposals": [item["proposal_id"] for item in proposals if item.get("can_materialize", True)],
        }

    def create_task_from_proposal(self, proposal_id: str) -> Dict[str, Any]:
        report = self.report()
        proposal = next(
            (item for item in report.get("evolution_proposals", []) if item.get("proposal_id") == proposal_id),
            None,
        )
        if not proposal:
            raise KeyError(proposal_id)
        objective = f"执行自进化提案 {proposal_id}：{proposal['title']}。验证标准：{proposal['validation']}"
        context = {
            "source": "evolution_harness",
            "proposal_id": proposal_id,
            "proposal": proposal,
            "risk_topics": ["harness", "evaluation", "tool", "memory"],
            "risk_level": "high" if proposal.get("priority") == "high" else "medium",
            "audit_item": "AuditPilot Agent 自进化链路",
        }
        return self.agent_runtime.create_task(objective, context)

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
                    "reason": "工具成功率低于 85%，需要进入失败归因与重试策略调参。",
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
        if float(observability.get("p95_latency_ms") or 0) > 5000:
            risks.append(
                {
                    "type": "latency_regression",
                    "severity": "medium",
                    "signal": f"p95_latency_ms={observability.get('p95_latency_ms')}",
                    "reason": "P95 延迟过高，建议拆分慢工具、开启缓存或使用异步执行。",
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
        backlog: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        proposals = [
            {
                "proposal_id": "HNS-01",
                "priority": "high" if backlog else "medium",
                "title": "把失败反思转成可回归的审计任务样例",
                "trigger": f"{observability.get('reflections', 0)} 条 runtime reflection；{len(backlog)} 个 benchmark backlog",
                "action": "从 needs_review / blocked / latency_regression 轨迹中抽取 objective、上下文、期望证据，写入下一轮 agent / rag evaluation cases。",
                "validation": "新增样例后执行 /api/training/evaluate 与 /api/evaluation/rag，release_gate 不新增 blocker。",
                "impact": "提升长周期审计任务稳定性，符合 AgentOps 与 self-evolving harness 方向。",
                "can_materialize": True,
            },
            {
                "proposal_id": "HNS-02",
                "priority": "medium",
                "title": "建立 Tool Calling 准确率与熔断恢复 playbook",
                "trigger": f"tool_success_rate={observability.get('tool_success_rate', 0)}；open_circuits={skill_metrics.get('open_circuits', 0)}",
                "action": "按 skill 汇总 error_type、input_schema、duration_ms，生成最小复现实例与恢复建议。",
                "validation": "同输入重复运行时成功率提升，cache / circuit 指标不退化。",
                "impact": "对齐大厂 JD 中工具调用准确性、失败定位和工程治理要求。",
                "can_materialize": True,
            },
            {
                "proposal_id": "HNS-03",
                "priority": "medium",
                "title": "沉淀审计经验单元，驱动跨会话自进化",
                "trigger": f"sessions={memory_stats.get('sessions', 0)}；episodes={memory_stats.get('episodes', 0)}",
                "action": "把高质量整改建议、证据缺口、控制测试模板沉淀为 profile / episode memory，并在相似审计场景自动召回。",
                "validation": "相同审计域二次提问时，答案包含历史标准、系统、风险主题且不引入冲突事实。",
                "impact": "对齐选择性记忆、经验单元沉淀与复杂 Agent 泛化研究。",
                "can_materialize": True,
            },
            {
                "proposal_id": "HNS-04",
                "priority": "medium",
                "title": "引入 role-level trace，让多 Agent 协作可单独评测",
                "trigger": "协作链已包含 planner / memory / research / evidence / control / risk / verifier。",
                "action": "为每个 agent_role 输出输入、决策、证据、失败模式和产物引用，支持角色级 pass / review / blocked。",
                "validation": "聊天与审计任务的 routing.agents 数量、role trace 完整度和 verifier 结论可被 UI 展示。",
                "impact": "让多 Agent 不只是展示标签，而是可观测、可评测、可复盘。",
                "can_materialize": True,
            },
        ]
        if risks:
            proposals.insert(
                0,
                {
                    "proposal_id": "HNS-00",
                    "priority": "high",
                    "title": "优先处理当前回归风险",
                    "trigger": f"{len(risks)} 个退化 / 阻断信号",
                    "action": "先关闭 release_gate blocker，再允许新能力合入，避免 harness 更新带来负收益。",
                    "validation": "所有 high 风险转为 closed，最近一次评测无 baseline regression。",
                    "impact": "符合 self-evolution 的风险优先原则。",
                    "can_materialize": True,
                },
            )
        if not eval_runs:
            proposals.append(
                {
                    "proposal_id": "HNS-05",
                    "priority": "high",
                    "title": "创建首个 Agent Benchmark 基线",
                    "trigger": "尚无持久化评测记录",
                    "action": "运行默认 agent benchmark，保存 baseline，后续所有优化都与该 baseline 比较。",
                    "validation": "data/evaluation_runs 生成 baseline_created 记录。",
                    "impact": "避免只有 demo，没有可量化演进证据。",
                    "can_materialize": True,
                }
            )
        return proposals

    def _harness_loops(self, proposals: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "loop": "Observe",
                "description": "从 evaluation run、runtime reflection、skill run log、memory stats 中收集可量化信号。",
                "artifacts": ["data/evaluation_runs", "data/agent_runtime", "data/skill_runs.jsonl"],
            },
            {
                "loop": "Mine Weakness",
                "description": "把 release gate、慢调用、熔断、低置信度和人工复核信号转为 benchmark backlog。",
                "artifacts": ["regression_risks", "benchmark_backlog"],
            },
            {
                "loop": "Propose",
                "description": "生成最小可验证改动：prompt、tool schema、memory、routing、eval case 或 UI smoke。",
                "artifacts": [proposal["proposal_id"] for proposal in proposals],
            },
            {
                "loop": "Validate",
                "description": "通过 baseline regression、质量门、人工复核和浏览器 smoke test 验证无负收益。",
                "artifacts": ["/api/training/evaluate", "/api/evaluation/rag", "/api/agent/observability"],
            },
            {
                "loop": "Materialize",
                "description": "把提案转成 AgentRuntime 任务，保留执行轨迹、工具调用、产物和反思。",
                "artifacts": ["/api/agent/evolution/proposals/{proposal_id}/task"],
            },
        ]

    def _maturity_score(
        self,
        coverage: Dict[str, Any],
        risks: List[Dict[str, Any]],
        observability: Dict[str, Any],
        memory_stats: Dict[str, Any],
        market: Dict[str, Any],
    ) -> int:
        score = 52
        score += int(coverage.get("coverage_rate", 0) * 18)
        score += int(market.get("alignment_rate", 0) * 14)
        if observability.get("tool_calls"):
            score += min(8, int(float(observability.get("tool_success_rate") or 0) * 8))
        if observability.get("reflections"):
            score += 4
        if memory_stats.get("sessions"):
            score += 4
        score -= sum(8 if item["severity"] == "high" else 4 for item in risks)
        return max(0, min(100, score))

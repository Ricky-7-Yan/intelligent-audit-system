# 10-Agent Harness 与自进化市场雷达

更新时间：2026-07-10

## 01 本轮升级目标

本轮升级围绕当前市场对 Agent 工程化的关注点：AgentOps、Harness、MCP / Tool Use、多 Agent 编排、选择性记忆、自进化经验沉淀，以及行业可交付工作台。升级后的 `EvolutionHarness` 不再只是静态 JD 对齐说明，而是可以读取运行时信号、生成 benchmark backlog、识别回归风险、输出自进化提案，并把提案转成可执行的 AgentRuntime 任务。

## 02 参考方向

- OpenAI Agents SDK 强调 Agent loop、handoffs、guardrails、function tools、MCP、sessions、human-in-the-loop、tracing 和 evaluation，这对应本项目的 `AgentRuntime`、`SafetyGate`、`SkillRegistry`、`ConversationMemory` 与 `EvaluationRunRepository`。
- MCP 将 AI 应用连接数据源、工具和工作流标准化，适合本项目继续从本地 SkillRegistry 扩展到真实 MCP server。
- AutoGen AgentChat 强调 multi-agent applications、agents、teams、memory/RAG、GraphFlow、tracing/observability，和本项目的多角色审计链路高度一致。
- LangSmith Observability 强调 trace、生产指标、监控、automations、feedback 与在线评测，对应本项目的 AgentOps / release gate / benchmark backlog。

## 03 已落地能力

### Market Radar

`/api/agent/evolution` 现在返回 `market_radar`，包含：

- AgentOps：生产轨迹、评测与回归闭环
- Harness / Sandbox：可控执行环境与长周期任务验证
- MCP / Tool Use：工具契约、权限与上下文装载
- 多 Agent 编排：角色分工、协作计划与人工复核
- 选择性记忆：经验单元、跨会话召回、污染控制
- 行业工作台：从聊天转向可交付流程

### Trajectory Protocol

新增 `audit-agent-trajectory-v2` 协议描述，要求任务轨迹至少包含：

- `task_id`
- `agent_role`
- `step_id`
- `skill`
- `status`
- `duration_ms`
- `attempts`
- `safety_gate`
- `reflection`
- `artifact_refs`

这让项目从“能跑任务”升级到“能复盘、能评测、能对齐生产治理”。

### Benchmark Backlog

Harness 会根据以下信号生成 backlog：

- release gate review / blocked
- tool success rate 下降
- skill circuit open
- P95 latency regression
- 缺少 baseline
- cache 命中不足

这些 backlog 可转成下一轮 agent / rag / runtime 评测用例。

### Evolution Control Plane

新增控制面：

1. Observe：读取 eval、runtime、skill、memory 信号
2. Mine Weakness：识别弱点与回归风险
3. Propose：生成最小可验证改动
4. Validate：通过 baseline regression 和 release gate 验证
5. Materialize：把提案转成 AgentRuntime 任务

### 提案转任务 API

新增接口：

```text
POST /api/agent/evolution/proposals/{proposal_id}/task
```

验证结果示例：

```json
{
  "success": true,
  "task_id": "AGT-4B50AE354B",
  "status": "running",
  "plan_steps": 7,
  "tool_calls": 1,
  "source": "evolution_harness",
  "proposal_id": "HNS-00"
}
```

## 04 面试讲述角度

这个项目现在可以这样讲：

> 我不是只做了一个审计聊天 Agent，而是做了一个面向企业审计交付的 Agent 工作台。它有多 Agent 路由、RAG、工具注册表、安全门禁、运行时轨迹、评测发布门禁和自进化 Harness。Harness 会持续读取失败、延迟、熔断、评测退化和记忆状态，自动产出回归用例候选和优化提案，并能把提案转成可执行任务，形成 Observe → Mine Weakness → Propose → Validate → Materialize 的闭环。

## 05 后续可继续增强

- 接入真实 MCP server，把 SkillRegistry 升级为本地 + 远程工具混合注册。
- 增加 role-level trace，让 planner / research / control / verifier 每个角色都可单独打分。
- 把 benchmark backlog 写入持久化数据集，并支持一键进入 `/training` 回归评测。
- 给 memory 增加经验单元评分、过期策略、冲突检测和隐私标签。
- 增加 sandbox worker，把工具执行从本地进程升级为隔离环境。

## 06 参考链接

- OpenAI Agents SDK: https://openai.github.io/openai-agents-python/
- Model Context Protocol: https://modelcontextprotocol.io/docs/getting-started/intro
- Microsoft AutoGen AgentChat: https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/index.html
- LangSmith Observability: https://docs.langchain.com/langsmith/observability

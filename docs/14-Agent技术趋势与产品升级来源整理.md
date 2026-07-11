# 14 Agent 技术趋势与产品升级来源整理

> 本文为内部参考文档，不写入客户使用的网站页面。更新时间：2026-07-11。

## 01 本轮查询来源

| 来源 | 链接 | 关键信号 | 对本项目的启发 |
| --- | --- | --- | --- |
| OpenAI Agents SDK 文档 | https://openai.github.io/openai-agents-python/ | Agent、Tools、Handoffs、Guardrails、Sessions、MCP、Tracing、Sandbox Agents | 继续强化 AgentRuntime、工具治理、会话记忆、审计任务可追踪和安全门禁 |
| Model Context Protocol 规范 | https://modelcontextprotocol.io/specification/2025-06-18 | MCP 用 JSON-RPC 连接 Host/Client/Server，能力包括 Resources、Prompts、Tools，并强调用户同意、隐私和工具安全 | 当前 Skill Registry/MCP-style tools 要继续向标准化 schema、权限、日志、确认机制演进 |
| LangChain State of AI Agents 报告 | https://www.langchain.com/stateofaiagents | Agent 生产落地比例提升；企业关注 tracing/observability、guardrails、offline evaluation、human oversight、工具权限 | 项目应突出评测门禁、运行时观测、人工复核、记录治理、全局检索与可解释执行轨迹 |
| Anthropic Building Effective Agents | https://www.anthropic.com/engineering/building-effective-agents | 成功 Agent 实现更偏简单、可组合模式；优先选择最简单方案；Agent 复杂性会换来延迟和成本；基础能力包括 retrieval、tools、memory | 审计 Agent 应保持“场景模板 + 可追踪工作流 + 必要 Agentic 能力”，避免为了炫技堆复杂度 |

## 02 技术趋势归纳

### 2.1 Tool Use / MCP 标准化

市场上对“能调用工具”已经不满足，关注点转向：

- 工具 schema 是否严格。
- 工具权限是否清晰。
- 调用前后是否有审计日志。
- 失败是否可重试、可熔断、可回放。
- 是否支持 MCP 等标准协议接入外部系统。

本项目已具备：

- `SkillRegistry`
- MCP-style tool 描述
- 权限注解
- 缓存、熔断、执行日志
- 全局搜索入口快速定位工具运行记录

后续可继续增强：

- tool selection precision 指标。
- 工具调用失败根因分类。
- 高风险写操作的人机确认机制。

### 2.2 Harness / Evaluation / Release Gate

企业真实落地 Agent 时，核心风险不是“能不能回答”，而是：

- 回答是否稳定。
- 工具调用是否可靠。
- RAG 是否可回溯。
- 变更是否引入回归。
- 高风险结论是否经过人工复核。

本项目已具备：

- Agent 评测
- RAG 评测
- Deep Research 评测
- 发布门禁
- 评测历史删除与折叠管理
- 自进化提案物化为运行任务

后续可继续增强：

- Bad Case 自动沉淀。
- 回归样本分层。
- 失败归因：规划失败、检索失败、工具失败、证据不足、记忆冲突。

### 2.3 Observability / Trace / Replay

Agent 越接近生产，越需要：

- 运行轨迹。
- 工具调用日志。
- 人工复核记录。
- 指标看板。
- 可定位历史记录。

本轮已落地：

- `GET /api/search`
- 全局命令中心 `Ctrl/⌘ + K`
- 审计记录、评测记录、证据分析、运行任务、工具日志聚合检索
- 搜索结果深链接打开审计、证据、评测和运行时任务

### 2.4 简单可组合，而不是过度复杂

Anthropic 的工程经验强调“先用简单、可组合模式解决问题”，这与本项目定位一致：

- 面向审计交付，不是泛聊天 Demo。
- 保留固定审计流程、控制矩阵、证据请求、人工复核。
- 在必要位置引入 Agentic 能力，如多 Agent 协作、工具调用、RAG、Harness。

## 03 本轮产品升级映射

| 市场关注点 | 本轮产品升级 |
| --- | --- |
| 复杂系统需要快速定位 | 新增全局命令中心 |
| 记录越多越要可治理 | 搜索聚合审计、评测、证据、任务、工具日志 |
| 可观测性不仅是看板，也要能跳转 | 搜索结果支持深链接打开对应记录 |
| 成熟产品需要快捷键和低摩擦入口 | 支持 Ctrl/⌘ + K 唤起 |
| 客户网站不能展示内部趋势研究 | 本文仅作为内部文档，不进入产品页面 |

## 04 后续建议

1. 给全局命令中心增加最近访问记录。
2. 给搜索结果增加“复制 ID”“删除记录”“导出报告”等上下文动作。
3. 把搜索结果和权限系统结合，高风险动作需要确认。
4. 为 MCP 工具增加真实 Server 接入示例。
5. 将 Bad Case 与全局搜索打通，支持按失败原因检索。

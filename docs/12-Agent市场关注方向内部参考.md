# 12 Agent 市场关注方向内部参考

> 本文是项目设计与面试表达的内部参考，不直接展示在客户产品网站中。客户界面只呈现真实可用的运行治理、任务执行、证据交付和记录管理能力。

## 01 记录目的

用户要求持续参考当前市场对 Agent 项目的关注方向，包括大厂 Agent 岗位 JD、人才计划、Harness、自进化、Tool Use、MCP、评测与 AgentOps 等内容。为了避免客户界面冗杂，相关调研沉淀在文档中，作为后续项目迭代和简历面试表达依据。

## 02 重点关注方向

| 方向 | 市场关注点 | 本项目对应实现 | 后续增强 |
| --- | --- | --- | --- |
| AgentOps | 生产轨迹、运行监控、失败归因、版本回归 | Agent Runtime、Skill 日志、Evaluation Repository、Release Gate | 失败自动聚类、版本趋势图、回滚建议 |
| Harness / Sandbox | 长周期任务执行环境、可控工具调用、隔离运行、预算控制 | SafetyGate、Skill Registry、任务封装、运行时指标 | 隔离 worker、权限分级、跨任务预算控制 |
| MCP / Tool Use | 工具协议、工具 schema、权限注解、工具选择准确率 | MCP-style tools、inputSchema、permissions、缓存、熔断 | 接入真实 MCP server，记录 tool selection precision |
| 多 Agent 协作 | Planner、Researcher、Verifier、Human-in-loop 分工 | Intent Router、agent_role、运行时 plan、复核闭环 | role-level trace 可视化与单独评分 |
| 自进化 | 从失败样例、评测退化、人工反馈中形成下一轮优化 | EvolutionHarness、Benchmark Backlog、提案物化任务 | Backlog 持久化并一键进入评测 |
| 选择性记忆 | 经验单元、相似任务召回、记忆污染和隐私控制 | ConversationMemory、session profile、episode memory | 经验评分、过期策略、冲突检测、隐私标签 |
| 端到端评测 | AgentBench / GAIA / TAU-bench 风格任务、RAG 与工具调用评测 | Agent 评测、RAG 评测、Deep Research 评测、Release Gate | 增加跨场景公开 benchmark 风格样例 |
| 垂直行业落地 | 不止聊天，要有业务工作台、审批、交付包和审计轨迹 | 审计项目、控制库、证据分析、整改闭环、报告下载 | 企业系统连接器、项目组合视图 |

## 03 与字节 / 腾讯 / 阿里 JD 的对应表达

### 字节跳动方向

- Harness Engineering、Agent Systems、长周期任务、工具环境、AI Coding / Search Agent。
- 本项目表达：
  - 有 Agent Runtime 和可执行任务轨迹。
  - 有 Tool Use 治理、缓存、熔断、安全门禁。
  - 有评测仓库和自进化提案闭环。

### 腾讯方向

- 多 Agent 协作、评测平台、工具调用准确性、失败归因、经验单元。
- 本项目表达：
  - 有多角色协作链和 role-level plan。
  - 有 Agent/RAG/Research 评测与 release gate。
  - 有 ConversationMemory 和可沉淀经验单元的路径。

### 阿里巴巴方向

- Agent 全生命周期、Planning、RAG、Function Calling / MCP、工程化落地。
- 本项目表达：
  - 审计工作台覆盖规划、取证、控制测试、发现、整改、交付。
  - Skill Registry 可复用为工具协议层。
  - FastAPI + 持久化仓库 + 前端控制台支撑产品化交付。

## 04 网站展示边界

客户网站不直接展示以下内容：

- 大厂 JD 名称和招聘导向。
- 市场趋势调研说明。
- 面试表达话术。
- 过多内部技术路线解释。

客户网站应展示：

- 当前任务状态。
- 工具调用和安全门禁。
- 优化队列和可执行提案。
- 证据、控制、报告和整改交付。
- 记录查看、折叠、删除和治理能力。

## 05 后续迭代提醒

1. 市场调研继续放在文档，不塞进客户 UI。
2. 客户 UI 优先呈现“我能用它完成什么”。
3. 技术亮点通过运行时证据体现，而不是把 JD 关键词贴在页面上。
4. 自进化能力要继续从“提案”走向“回归验证、版本对比、自动回滚建议”。

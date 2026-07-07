# 2026 大厂 Agent 岗位 JD 来源整理

整理日期：2026-07-07

## 1. 检索关键词

- ByteDance Agent Systems Harness Engineering Model Self-improvement Skills Knowledge Base
- 字节 Seed Code Agent Test-time Scaling Long Horizon Task Multi-Agent RL
- 腾讯 AI Agent 评测 多轮对话 工具调用 根因分析 CodeBuddy
- 腾讯 青云计划 大模型 智能体 强化学习 Agent 前沿评测 经验单元
- 阿里巴巴 AI Agent RAG Function Calling MCP Prompt 工程化 SFT RL
- 淘天 AI Agent 优化工程师 Agent 编排 任务规划 Prompt 工程化

## 2. 来源列表

| 公司/方向 | 参考链接 | 主要能力信号 | 项目映射 |
| --- | --- | --- | --- |
| ByteDance Seed / Agent Systems | `https://joinbytedance.com/search/7610602401397639477` | Agent Systems、AI Coding Environment、Harness Engineering、Skills、知识库、Model Self-improvement | Agent Runtime、Skill Registry、Evaluation Repository、Evolution Harness |
| ByteDance Experienced Position | `https://jobs.bytedance.com/experienced/position/7647554140894185781/detail` | Agent 研发、Planning、Memory、Reflection、Multi-Agent、Tools、RAG | Intent Router、Conversation Memory、Agentic RAG、运行时反思 |
| Tencent Careers | `https://careers.tencent.com/` | AI Agent 测试、评测平台、多轮对话质量、工具调用准确性、失败归因 | Evaluation runs、release gate、runtime observability、reflection issues |
| Tencent 青云计划相关公开信息 | 腾讯招聘与公开人才计划页面 | 智能体、强化学习、Agent 前沿评测、经验单元沉淀、协同进化 | Conversation Memory、Self-Evolution Harness、JD coverage |
| Alibaba Talent | `https://talent.alibaba.com/` | 大模型算法、AI Agent、RAG、Function Calling、Prompt 工程化、Agent 编排 | Skill schema、MCP-style tools、RAG evaluator、审计全生命周期 |
| Alibaba Campus Talent | `https://campus-talent.alibaba.com/` | 校招/人才计划，大模型、智能体、训练/评测、业务落地 | 训练评测文档、项目归档、README 面试表达 |

## 3. 与项目新增能力的对应关系

### Harness Engineering

项目已实现：

- 持久化 Agent task。
- Skill 输入 schema。
- 工具超时、缓存、熔断。
- baseline 对比与 release gate。
- 自进化建议接口。

后续可选：

- Docker sandbox。
- 更长周期任务队列。
- 可视化 failure trace 和 replay。

### Model Self-improvement / Self-Evolution

项目已实现：

- Runtime reflection。
- Evaluation regression detection。
- Self-Evolution Harness proposal。
- Memory profile 与 episode 沉淀。

后续可选：

- 从 failure case 自动生成训练样本。
- 引入 LLM-as-Judge 多维 rubric。
- 对 prompt、tool schema、retrieval strategy 做候选修改 A/B 验证。

### Multi-Agent 与 Context Management

项目已实现：

- HybridIntentRouter 输出审计意图、置信度、实体、Agent 列表。
- ConversationMemory 提供 working/episodic/profile memory。
- Chat API 把 routing 与 memory 一起返回给前端。

后续可选：

- 多 Agent 并行执行与冲突仲裁。
- 跨项目审计策略召回。

### RAG / Function Calling / MCP

项目已实现：

- Agentic RAG pipeline。
- RAG evaluator。
- MCP-style tool descriptors。
- SkillRegistry schema 与 permission。

后续可选：

- 真实 MCP server 接入。
- 企业内部知识库 connector。

## 4. 面试使用提示

建议在简历项目描述中使用以下关键词：

- Agentic RAG
- Multi-Agent Routing
- Long-term Memory
- Tool Calling Governance
- MCP-style Skill Registry
- Reflection and Release Gate
- Self-Evolution Harness
- Baseline Regression Evaluation
- Human-in-the-loop Audit Delivery


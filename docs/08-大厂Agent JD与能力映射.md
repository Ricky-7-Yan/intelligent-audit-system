# 08 大厂 Agent JD 与能力映射

本文整理 2026 年 7 月 7 日补充检索到的字节跳动、腾讯、阿里巴巴及相关人才计划/行业 Agent 岗位信号，并把要求映射到审脉 AuditPilot 的真实实现。

## 1. 结论

审脉 AuditPilot 当前已经覆盖大厂 Agent 岗位最看重的六类能力：

1. Agent 核心架构：Planning、Memory、Reflection、Multi-Agent、Skills/Tools、Agentic RAG。
2. 工程化落地：FastAPI 服务、持久化任务、Skill schema、工具超时/缓存/熔断、质量门、审计交付包。
3. 评测与发布：Agent/RAG 评测、baseline 对比、release gate、退化风险识别。
4. Self-Evolution / Harness：从运行时反思、评测失败、Skill 指标和 JD 能力项生成下一轮优化建议。
5. 企业审计业务闭环：审计范围、控制矩阵、证据请求、控制测试、整改任务、人工复核。
6. 产品化界面：工作台、审计项目、Agent 协作、知识与证据、运行时、评测与发布。

## 2. 网上 JD 与人才计划摘录

### 2.1 字节跳动 / Seed / Agent Systems

来源参考：

- ByteDance / Seed Agent Systems 页面：`https://joinbytedance.com/search/7610602401397639477`
- ByteDance 职位页搜索结果：`https://jobs.bytedance.com/experienced/position/7647554140894185781/detail`
- ByteDance Seed / Agent Systems 相关公开搜索结果，关键词包括 `Agent Systems`、`AI Coding Environment`、`Harness Engineering`、`Model Self-improvement`。

JD 信号：

- 要求建设 Agent 执行环境、Harness Engineering、命令行/IDE/浏览器/沙箱等环境集成。
- 关注 Skills、知识库、多模态能力与端到端 AI Coding Agent 的效果提升。
- 强调 Model Self-improvement、基准评测、复杂任务执行和长周期能力。
- 对研究型岗位，常出现 Code Agent RL、Test-time Scaling、Long Horizon Task、Multi-Agent RL 等方向。

映射到本项目：

- `services/agent_runtime.py`：A2A 风格任务信封、计划步骤、工具调用、反思、重试、观测指标。
- `services/skill_registry.py`：MCP 风格 Skill 描述、输入 schema、超时、TTL cache、失败阈值、熔断。
- `services/evaluation_repository.py`：baseline 对比、release gate、回归阻断。
- `services/evolution_harness.py`：把运行失败、评测退化、记忆状态和 JD 能力项汇总成自进化建议。

仍可继续增强：

- 将当前本地执行环境升级为 Docker/Kubernetes worker 沙箱。
- 接入更长周期的任务 benchmark，例如跨多轮审计项目和证据追踪。
- 引入真实模型微调/RL 训练数据，而不只做工程侧 Harness。

### 2.2 腾讯 / 青云计划 / AI Agent 评测与平台方向

来源参考：

- 腾讯招聘搜索：`https://careers.tencent.com/`
- 腾讯云 AI Agent 测试工程师相关公开搜索结果。
- 腾讯青云计划相关公开信息，关键词包括 `大模型`、`智能体`、`强化学习`、`Agent 前沿评测`、`经验单元`、`协同进化`。

JD 信号：

- 要求多 Agent 协作、复杂任务拆解、Context Management、工具调用准确性。
- 关注 Agent 评测平台、任务完成率、多轮对话质量、失败归因和根因分析。
- 人才计划中强调智能体、强化学习、开放域任务协同进化、经验单元沉淀。

映射到本项目：

- `services/intent_router.py`：审计意图识别、实体抽取、Agent 路由和 multi-agent 触发。
- `services/conversation_memory.py`：工作记忆、episodic memory、profile memory、相似历史召回。
- `services/evolution_harness.py`：把 reflection、evaluation、Skill failure 转换为候选改进。
- `/api/agent/observability` 与 `/api/agent/evolution`：运行时观测和自进化建议。

仍可继续增强：

- 引入公开 AgentBench/GAIA/TAU-bench 风格样例作为横向评测集。
- 对每类失败做更细粒度 root cause 分类，例如“检索失败 / 工具 schema 错误 / 规划过粗 / 记忆冲突”。

### 2.3 阿里巴巴 / 通义 / 淘天 / AI Agent 方向

来源参考：

- 阿里巴巴招聘与校园招聘：`https://talent.alibaba.com/`、`https://campus-talent.alibaba.com/`
- 阿里 AI Agent / 大模型算法 / AI Agent 优化工程师相关公开搜索结果，关键词包括 `Prompt 工程化`、`Agent 编排`、`Function Calling`、`MCP`、`RAG`、`SFT/RL`。

JD 信号：

- 强调 Agent 全生命周期、任务规划、多步推理、RAG、工具调用、端到端评测。
- 工程岗强调 Prompt 工程化、Agent 编排、Function Calling/MCP、业务落地。
- 算法岗会要求 SFT/RL、后训练、复杂场景数据构造和效果评测。

映射到本项目：

- 审计全生命周期：立项、范围、证据、控制测试、发现、整改、交付。
- `SkillRegistry` 与 `/api/mcp/tools`：可扩展工具协议层，便于迁移到 MCP server。
- `RAGEvaluator` 与 `EvaluationRunRepository`：检索增强问答和 Agent 输出的可回归评测。
- 产品界面把能力拆成“Agent 协作 / 知识与证据 / 运行时 / 评测与发布”，面试展示更像真实产品。

仍可继续增强：

- 与真实 MCP server、企业知识库、审计底稿系统做深度对接。
- 增加面向特定业务域的 SFT 数据构造脚本和训练流水线。

## 3. 能力矩阵

| JD 能力项 | 项目实现 | 证据文件/接口 | 当前状态 |
| --- | --- | --- | --- |
| Planning / 任务拆解 | Agent Runtime 自动生成审计计划、控制映射、证据清单、整改步骤 | `services/agent_runtime.py`、`/api/agent/tasks` | 已实现 |
| Memory / 上下文管理 | 工作记忆、历史压缩、profile、相似历史召回 | `services/conversation_memory.py`、`/api/memory/sessions` | 已实现 |
| Reflection / 自反思 | 每个步骤生成 confidence、issues、next_action | `services/agent_runtime.py` | 已实现 |
| Tool Use / Function Calling | Skill schema、权限、超时、缓存、熔断、运行日志 | `services/skill_registry.py`、`/api/skills` | 已实现 |
| MCP 风格扩展 | 工具描述、输入 schema、权限声明、MCP tools API | `/api/mcp/tools` | 已实现 |
| Agentic RAG | 查询扩展、检索、引用、降级、RAG 评测 | `rag/`、`services/rag_evaluator.py` | 已实现 |
| Multi-Agent | Router 输出 planning/control/evidence/remediation 等角色 | `services/intent_router.py`、`/api/agent/route` | 已实现 |
| Evaluation Harness | Agent/RAG run、baseline、release gate、退化阻断 | `services/evaluation_repository.py` | 已实现 |
| Self-Evolution Harness | JD 覆盖、运行时信号、风险、下一轮优化建议 | `services/evolution_harness.py`、`/api/agent/evolution` | 已实现 |
| 企业级安全治理 | SafetyGate、人工复核、审计交付可追溯 | `services/safety_gate.py`、审计 run review APIs | 已实现 |
| 产品化交互 | 工作台、审计项目、Agent 协作、知识证据、运行时、评测发布 | `templates/`、`static/app.css` | 已实现 |

## 4. 面试表达建议

可以把项目概括为：

> 我做的是一个企业审计场景的 AgentOps 产品，不只是 RAG Demo。它有 Agentic RAG、意图路由、多 Agent 协作、跨会话记忆、Skill 工具治理、运行时反思、评测 baseline 和 release gate。为了对齐大厂 Agent 岗，我又补了 Self-Evolution Harness：系统会从运行失败、评测退化、Skill 熔断和 JD 能力项里生成下一轮可验证优化建议。

字节侧重点：

- 强调 Harness Engineering、执行环境、Skill/工具集成、Self-improvement 和长周期任务。

腾讯侧重点：

- 强调多 Agent 协作、Context Management、评测平台、失败归因、经验单元沉淀。

阿里侧重点：

- 强调业务落地、Prompt/Agent 编排、RAG、Function Calling/MCP、端到端评测和全生命周期交付。


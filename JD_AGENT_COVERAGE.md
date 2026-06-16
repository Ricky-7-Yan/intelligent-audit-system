# 审脉 AuditPilot：Agent 岗位能力覆盖说明

本文件用于项目复盘、简历和面试讲解，不面向客户界面展示。能力规划参考用户提供的字节 Seed 搜索问答 Agent JD，并结合 2026 年公开招聘信息中高频出现的 Agent Runtime、Tool/Memory/Context、MCP/Skill、Deep Research、评测闭环和安全治理要求。

## 已覆盖能力

### 1. 搜索问答与 Deep Research

- `services/research_agent.py` 支持意图识别、查询改写、多源检索融合、推理轨迹、证据缺口和答案质量评估。
- `rag/agentic_rag.py` 支持持久化知识库、种子知识、语义检索降级、关键词检索、来源引用和 RAG 评估。
- `/api/research/answer`、`/api/evaluation/rag`、`/api/research/evaluation-plan` 提供 Deep Research 与评测入口。

### 2. Agent Runtime 与任务协议

- `services/agent_runtime.py` 新增 `audit-agent-task-v1` 任务包，包含 `objective`、`context`、`plan`、`steps`、`artifacts`、`tool_calls`、`safety_gate`、`metrics`。
- `/api/agent/tasks` 支持创建持久化任务；`/api/agent/tasks/{task_id}/run-next` 支持逐步执行；`/api/agent/observability` 输出运行时观测指标。
- 任务执行会自动调用 Skill Registry，形成可复盘的计划、工具调用和产物链路。

### 3. Tool Use、MCP 与 Skill 治理

- `services/skill_registry.py` 提供 Skill 注册、输入 Schema、权限声明、版本、MCP 风格工具描述和运行日志。
- `/api/skills`、`/api/mcp/tools`、`/api/skills/runs`、`/api/skills/metrics` 覆盖工具发现、工具调用、日志和指标。
- Skill 覆盖审计范围规划、控制矩阵映射、证据清单、发现草稿、RAG 查询、评测用例设计和整改任务生成。

### 4. 安全门禁与 Human-in-the-loop

- `services/safety_gate.py` 新增运行时安全检查，覆盖密钥泄漏、破坏性动作、证据不足等风险。
- `/api/safety/check` 提供独立安全检查接口；Agent Runtime 每个任务和步骤都会执行安全门禁。
- 审计工作台保留人工复核、补证、退回和审批闭环，避免无证据自动下结论。

### 5. 评测体系与工程闭环

- `/training` 已升级为 Agent 评估与发布门禁工作台，覆盖 Agent、RAG、Deep Research、工具轨迹、真实性、权威性、相关性和用户体验。
- `services/evaluation_repository.py` 持久化评测记录，便于形成版本回归和 badcase 闭环。
- Skill 与 Agent Runtime 增加成功率、平均耗时、P95 延迟、失败分布、输入/输出大小和成本占位指标。

### 6. 审计行业落地能力

- `/audit` 覆盖审计立项、范围、业务背景、重点问题、证据、控制测试、质量门、底稿索引、报告下载、交付包、人工复核和整改任务。
- `services/evidence_analyzer.py` 支持 CSV、JSON、日志和文本证据分析，生成字段画像、风险信号、控制映射和补证建议。
- `services/audit_delivery.py` 将审计运行、证据分析、控制测试和整改内容组织为交付包。

## 与公开 JD 要求的映射

- 腾讯元宝 Agent 架构类岗位强调 Agent Runtime、Tool/Memory/Context 抽象、多 Agent 协作、Human-in-the-loop 和在线系统架构，本项目已落地 Runtime、Tool 抽象、安全门禁、人工复核与持久化运行记录。
- 腾讯 Agent Evaluation 类岗位强调真实 Agent 系统的评估和可靠性基础设施，本项目已覆盖 Skill 指标、Agent 运行时观测、评测历史、发布门禁和 badcase 数据基础。
- 微信搜索/字节 Seed 搜索 Agent 类岗位强调 Search Agent、DeepSearch/DeepResearch、复杂任务、多轮推理、Context/Memory、自反思、工具学习和评测闭环，本项目已覆盖 Deep Research、RAG、推理轨迹、工具执行、证据缺口、评测和人工复核。
- 阿里 AI Agent / Infra / Skill 方向常见要求强调工程化 Agent 应用、工具平台化、稳定性、服务化和业务场景落地，本项目已覆盖 FastAPI 服务、Docker 部署、工具注册、观测指标和审计业务闭环。

## 后续可继续增强

- 增加真正的多 Agent 角色协同：Planner、Retriever、Controller、Reviewer、Reporter。
- 增加长期 Memory：把项目历史、复核意见、整改状态压缩为可检索记忆。
- 增加异步任务队列和 SSE 流式执行：提升复杂任务体验和并发能力。
- 增加权限体系：租户、角色、审计项目隔离和操作审计。
- 增加生产级数据库后端：将当前 JSONL/JSON 持久化升级为 PostgreSQL 或 MySQL。

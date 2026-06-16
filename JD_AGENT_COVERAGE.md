# 审脉 AuditPilot：Agent 岗位 JD 覆盖说明

本项目定位为面向审计行业真实交付的企业级 Agent 产品，而不是通用运营 Demo。能力规划参考用户提供的字节 Seed 搜索问答 Agent JD，并补充腾讯招聘官网 `careers.tencent.com/tencentcareer/api/post/Query` 在 2026-06-16 检索到的 Agent/大模型岗位。

## 腾讯官方 JD 要点

- Agent Evaluation Intern 107491，PostId `2057058794919346176`，2026-05-20 更新：要求自动化评估流水线、执行产物采集、工具调用评估、轨迹/日志/中间步骤分析、成功率/工具精度/恢复率/延迟/成本/安全失败率指标、合成用例/黄金流程/匿名真实执行数据集、badcase 到工程改进闭环。
- 元宝-Agent架构工程师，PostId `2016726997581058048`，2026-06-04 更新：要求 Agent Runtime、Tool/Memory/Context 抽象、多 Agent 协作、Human-in-the-loop，以及结合模型能力边界和架构约束做系统方案。
- 微信搜索-Agent算法专家，PostId `2062097072978575360`，2026-06-11 更新：要求 Search Agent、DeepSearch/DeepResearch、真实世界复杂任务 Agentic 能力、Context/Memory、Agentic RL、Agent 自进化和大模型结合搜索范式。
- 腾讯视频-AI Agent工程师，PostId `2049051430010122240`，2026-06-09 更新：要求任务规划、工具调用、记忆管理、多轮决策、高并发 Agent 服务框架、工作流引擎、Workflow/DAG/多 Agent 协作、Function Calling、Tool Use、RAG、上下文管理、效果评估和 badcase 迭代。

## 项目覆盖

### 1. 下一代搜索问答 Agent

- 意图理解：`services/research_agent.py` 对问题做场景分类和领域识别。
- 查询改写：同一审计问题生成多路查询，覆盖审计对象、标准、证据和控制测试语义。
- RAG：`rag/agentic_rag.py` 支持持久化知识库、种子知识、语义检索降级、关键词检索和来源引用。
- 多源融合：`AuditResearchAgent` 合并多查询来源并按分数去重排序。
- 答案生成：输出结论、依据、推理、建议动作和证据缺口。

### 2. Deep Research 与跨文档推理

- `/api/research/answer` 返回 `query_rewrites`、`sources`、`reasoning_trace` 和 `evaluation`。
- 面向复杂审计问题支持理解、检索、验证、决策四阶段处理。
- 对证据不足场景显式触发人工复核，不输出不可追溯结论。

### 3. Agentic 架构

- 审计工作流：规划、检索、控制映射、风险评分、审计程序、质量门、整改计划。
- Runtime 抽象：`AuditAgent` 维护会话 Memory、执行 trace 和质量门；`SkillRegistry` 提供 MCP-style schema、权限、版本和运行日志。
- Human-in-the-loop：审计工作台支持人工复核、补证、退回和审批。
- 多 Agent 演进空间：当前按工作流阶段分层，后续可扩展为 Planner、Retriever、Controller、Reviewer 多角色协作。

### 4. Tool Use 与 MCP/Skill

- `/api/skills` 和 `/api/mcp/tools` 暴露工具能力、输入 schema、权限和版本。
- Skill 覆盖风险评估、控制映射、证据请求、报告生成、Agent 评测用例设计等审计核心动作。
- 评估页新增 `tool_trace_quality`、`agentic_capability`、`human_review_awareness` 指标，检查工具轨迹是否完整。

### 5. 评测体系与发布门禁

- `/training` 已升级为 Agent 评估与发布门禁工作台，支持 Agent、RAG、Deep Research、JD 覆盖四类入口。
- `/api/training/evaluate` 支持自定义 golden case，并输出真实性、完整性、审计专业性、可执行性、合规对齐、Agentic 能力、轨迹质量和人工复核意识。
- `/api/evaluation/rag` 输出关键词覆盖、来源数量、权威性、检索置信度和失败模式。
- `/api/research/evaluation-plan` 定义真实性、时效性、权威性、相关性、工具调用、轨迹质量和用户体验指标。
- 评估结果包含回归风险和优化建议，可作为后续版本发布准入门禁。

### 6. 审计行业落地

- 审计模板覆盖 SOX ITGC、ERP 权限、数据安全、变更发布、备份恢复、第三方服务。
- 业务闭环覆盖审计档案、风险台账、证据请求中心、控制测试工作台、底稿索引、复核和整改跟踪。
- 交付包和报告支持审计交付物沉淀，不依赖单次聊天结果。

### 7. 后续强化方向

- 将每次评测结果持久化，形成模型/Prompt/知识库版本对比趋势。
- 为每个工具调用记录 input/output/error/retry/cost/latency，进一步覆盖腾讯 Agent Evaluation JD。
- 增加 Memory 摘要和长期项目上下文，将多轮审计会话纳入 Deep Research。
- 增加发布前自动化门禁脚本，把 critical regression 作为上线阻断条件。

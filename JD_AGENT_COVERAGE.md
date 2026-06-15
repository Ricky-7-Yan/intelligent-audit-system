# 审脉 AuditPilot：Agent 岗位 JD 覆盖说明

本项目按字节跳动招聘官网 Seed/搜索问答 Agent 相关岗位，以及用户提供的 Seed JD 要点进行能力映射。核心定位不是通用运营平台，而是面向审计行业的企业级 Agent 产品。

## 1. 下一代搜索问答 Agent

- 意图理解：`services/research_agent.py` 对问题做场景分类和领域识别。
- 查询改写：同一问题生成多路查询，覆盖审计对象、标准、证据和控制测试语义。
- 检索增强：`rag/agentic_rag.py` 支持持久化知识库、种子知识、语义检索降级、关键词检索和来源引用。
- 多源融合：`AuditResearchAgent` 合并多查询来源并按分数去重排序。
- 高质量答案：输出结论、依据、推理、建议动作和证据缺口。

## 2. Deep Research 与跨文档推理

- `/api/research/answer` 返回 `query_rewrites`、`sources`、`reasoning_trace` 和 `evaluation`。
- 面向复杂审计问题支持多步骤处理：理解、检索、验证、决策。
- 对证据不足场景显式触发人工复核，不输出不可追溯结论。

## 3. Reasoning、反思与 Agentic 能力

- 审计 Agent 工作流：规划、检索、控制映射、风险评分、审计程序、质量门、整改计划。
- 质量门：输出置信度、证据扎实度、控制覆盖和缺失证据。
- 反思验证：Research API 内置真实性、权威性、相关性、完整性和人工复核标记。
- Agentic 编排：Skill Registry 和 MCP-style tools 暴露 schema、权限、版本和运行日志。

## 4. 高价值行业场景落地

- 审计模板覆盖 SOX ITGC、ERP 权限、数据安全、变更发布、备份恢复、第三方服务。
- 业务闭环覆盖审计档案、风险台账、证据请求中心、控制测试工作台、底稿索引、复核和整改跟踪。
- 下载交付包和报告支持审计交付物沉淀。

## 5. 评测体系与工程闭环

- `/api/research/evaluation-plan` 定义真实性、时效性、权威性、相关性和用户体验指标。
- `/api/evaluation/rag` 保留 RAG 评测入口。
- `/api/product/overview` 汇总知识库、审计档案、风险、证据请求、控制健康和整改状态。
- 支持 DeepSeek/OpenAI-compatible LLM 网关，同时具备无模型降级运行能力，方便演示和工程部署。

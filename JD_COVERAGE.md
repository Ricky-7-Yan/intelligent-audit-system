# Agent 岗位能力覆盖说明

本项目面向大模型 Agent / AI 应用工程岗位展示，重点覆盖校招和人才计划中常见的工程要求。

## 能力映射

| 岗位能力 | 项目覆盖 |
| --- | --- |
| LLM 应用开发 | FastAPI + 可选 Qwen/OpenAI 兼容接口，支持 LLM 不可用时降级 |
| Agent 架构 | 任务规划、工具调用、证据检索、控制映射、质量门、人工复核 |
| RAG | 文档切块、持久化知识库、查询扩展、混合检索、来源引用 |
| RAG 评测 | `/api/evaluation/rag` 支持基准用例和自定义用例评估 |
| 工具使用 | MySQL、Neo4j、RAG、审计控制库、报告导出、任务状态更新 |
| 业务落地 | 审计程序、抽样计划、审计发现草稿、整改任务跟踪 |
| 工程化 | Dockerfile、docker-compose、健康检查、配置模板、git 版本管理 |
| 可观测与质量 | execution trace、quality gate、confidence、groundedness、control coverage |
| 数据安全 | 本地 `config.env`、模型、日志、运行数据均通过 `.gitignore` 保护 |

## 推荐演示路径

1. 打开 `/audit`，运行 “ERP系统权限管理 / 安全审计 / ISO27001”。
2. 展示 KPI、控制矩阵、质量门、证据包、审计程序和抽样计划。
3. 在“整改任务跟踪”中更新任务状态。
4. 提交人工复核意见。
5. 下载 Markdown 审计报告。
6. 打开 `/training` 运行 Agent 评估。
7. 调用 `/api/evaluation/rag` 展示 RAG 检索评估能力。

## 后续可扩展方向

- 接入真实企业权限、变更、日志数据源。
- 增加基于 OpenTelemetry 的链路追踪。
- 将整改任务同步到 Jira、飞书或企业微信。
- 增加更严格的 RAG faithfulness / answer relevance 评测。
- 增加多 Agent 协作：审计经理、证据检索员、控制测试员、复核员。

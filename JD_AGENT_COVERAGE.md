# 审脉 AuditPilot：大厂 Agent JD 能力覆盖

本文件记录截至 2026-07-06 的项目能力基线。更完整的简历与面试材料见 [docs/05-JD对齐与简历面试指南.md](docs/05-JD对齐与简历面试指南.md)。

## 核心映射

| 招聘高频要求 | 已落地实现 | 验证方式 |
| --- | --- | --- |
| Planning / Multi-step | 依赖计划、逐步执行、任务预算 | `/skills` 创建任务 |
| Memory / Context | 工作、情景、画像、相关历史 | `/chat` 连续对话 |
| Reflection | 每步反思、置信度、问题、下一动作 | Runtime 任务详情 |
| Multi-Agent | 融合路由选择多个领域 Agent | `POST /api/agent/route` |
| RAG / Knowledge | 切块、混合检索、查询扩展、来源与降级 | `/knowledge` |
| Skills / Tools / MCP | Schema、权限、版本、MCP 描述 | `/api/mcp/tools` |
| 工具可靠性 | 超时、TTL 缓存、熔断、重试、日志 | `/api/skills/runs` |
| Eval / Regression | Agent/RAG/Research、基线差异、发布门禁 | `/training` |
| Safety / HITL | 安全门、证据质量门、人工复核 | `/audit` |
| 业务落地 | 立项、证据、控制、发现、整改、交付包 | `/audit` |
| 工程化 | Python、FastAPI、Docker、OpenAPI、持久化 | `/docs`、部署文件 |

## 字节方向

公开岗位常强调 Planning、Memory、Reflection、多步推理、多 Agent、Skills/Tools、RAG、Self-evolving/Evaluator 和真实端到端系统。本项目已覆盖上述应用工程链，并通过审计交付场景提供可验证产物。

## 腾讯方向

公开岗位常强调复杂工作流、RAG/Function Calling、Agent Runtime、Tool/Memory/Context 抽象、可靠性和评测基础设施。本项目已落地融合路由、Runtime、工具治理、记忆、反思、观测和发布门禁。

## 阿里方向

公开岗位常强调大模型服务端、Prompt/模型编排、RAG/Multi-Agent、缓存、服务治理、容器化和业务产品化。本项目覆盖 Agent 应用服务、工具缓存/熔断、Docker、健康检查与完整审计业务闭环；多模型网关和分布式队列列入生产化演进。

## 不夸大的边界

- 未完成正式压测，不声明线上并发和可用性数字。
- 内置评测不等于真实客户准确率。
- 当前文件 Repository 是可运行实现，生产规模应迁移到数据库、对象存储和任务队列。
- 当前核心是 Agent 应用工程；SFT/RLHF 仅保留离线入口，不包装成已完成训练平台。

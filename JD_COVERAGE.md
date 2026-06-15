# Agent 岗位能力覆盖说明

本项目面向大模型 Agent / AI 应用工程岗位展示。能力设计参考了企业招聘官网上可见的人才计划与 Agent 相关岗位方向：

- 腾讯青云计划：官方页面强调面向顶尖技术人才，覆盖 AI、大模型、基础软件、云计算、安全等方向，重视工程落地与复杂问题解决。
- 字节跳动 Seed：校园招聘官网可见大语言模型、Code Agent、通用 Agent、算法与工程相关岗位方向，核心关键词包括 Agent、工具使用、代码/任务自动化、模型训练与评测。
- 阿里巴巴阿里星与校园招聘：招聘官网可检索到 AI Agent 优化工程师等方向，岗位名明确覆盖训练、数据、评测，强调 Agent 能力优化、数据闭环和工程交付。

说明：企业官网 JD 会随时间调整，本项目不复制完整 JD 文本，而是把真实岗位中反复出现的能力要求转化为可运行功能。

## 能力映射

| 岗位能力 | 项目覆盖 |
| --- | --- |
| LLM 应用开发 | FastAPI + 可选 Qwen/OpenAI 兼容接口，支持 LLM 不可用时确定性降级 |
| Agent 架构 | 任务规划、证据检索、控制映射、风险评分、质量门、报告生成、整改闭环 |
| Tool Calling | Skill Registry 暴露可调用工具，包含 schema、权限、版本和执行日志 |
| Skill 化能力 | 审计范围规划、证据清单、控制映射、发现草稿、整改计划、RAG 查询、评测用例设计 |
| MCP 风格协议 | `/api/mcp/tools` 输出工具名、描述、inputSchema、annotations，可供 Agent Client 发现 |
| RAG 正确实施 | 文档切块、持久化知识库、种子知识加载、查询扩展、混合检索、来源引用、置信度 |
| RAG 评测 | `/api/evaluation/rag` 支持默认与自定义用例，评估 term coverage、source coverage、confidence |
| 数据与评测闭环 | `/training` 页面运行 Agent/RAG 评估，`agent.eval_case_designer` 生成评测用例 |
| 企业业务落地 | SOX ITGC、ISO27001、COBIT、NIST、CIS、数据安全、ERP 权限、变更、日志、备份恢复 |
| 可观测性 | execution trace、quality gate、confidence、groundedness、control coverage、Skill run log |
| 人工复核 | 审计运行档案支持复核人、结论、意见、状态流转 |
| 工程化部署 | Dockerfile、docker-compose、健康检查、配置模板、Windows 一键启动 |
| 数据保护 | config/env、模型、运行日志、审计档案、上传文件通过 `.gitignore` 保护 |

## 展示路径

1. 打开 `/audit`，运行“ERP系统权限管理 / 安全审计 / ISO27001”。
2. 展示任务计划、RAG 证据包、控制矩阵、质量门、审计程序、抽样计划和审计发现草稿。
3. 在整改任务跟踪中更新状态、责任人和备注。
4. 提交人工复核意见并下载 Markdown 审计报告。
5. 打开 `/knowledge`，检索 SOX ITGC、NIST CSF、CIS Controls、数据安全或 Agent 平台问题。
6. 打开 `/training`，运行 Agent 评估与 RAG 评测。
7. 打开 `/skills`，展示 Skill Registry、MCP Tools 和 Skill 调用日志。

## 本轮增强点

- 新增 `data/seed_knowledge/enterprise_audit_seed.json`，覆盖 NIST、CIS、ISO27001、COBIT、SOX ITGC、数据安全、Agent 平台、Skill/MCP、RAG 评测和审计发现写作。
- RAG 启动时自动加载种子知识，不覆盖用户上传或运行产生的数据。
- Skill 从 4 个扩展到 7 个，新增控制映射、评测用例设计、整改任务生成。
- 模板生成器改为 UTF-8 + HTML entity 输出，降低 Windows 环境下中文乱码风险。
- 前端页面修复重复 HTML 尾部和脚本加载顺序问题。

## 后续可扩展方向

- 接入真实企业权限、变更、日志、工单、备份和漏洞数据源。
- 增加 OpenTelemetry 链路追踪与工具调用耗时统计。
- 对接真实 MCP server/client。
- 增加 faithfulness、answer relevance、context precision 等更严格 RAG 评测指标。
- 增加多 Agent 协作角色：审计经理、证据检索员、控制测试员、复核员。

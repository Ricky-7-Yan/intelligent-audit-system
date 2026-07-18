# 审脉 AuditPilot

面向真实审计交付的企业级 AI Agent 工作台。AuditPilot 将审计立项、RAG 取证、控制映射、风险评估、整改闭环、Human Review、评测门禁和交付包沉淀在同一条可追溯证据链里，用一个可运行的 Web 产品展示 Agent 工程化落地能力。

> 默认支持无外部大模型密钥的确定性降级模式；配置兼容 OpenAI 协议的模型后，可启用 LLM 增强分析。真实 API Key 只应放在本地 `config.env` 或部署平台 Secret 中，仓库不会提交密钥。

![AuditPilot 工作台](docs/screenshots/auditpilot-overview-desktop.png)

## 项目亮点

| 能力 | 说明 |
| --- | --- |
| 审计交付闭环 | 从范围、控制矩阵、审计程序、抽样、证据请求、发现、整改到交付包，形成可复核链路。 |
| Agent Runtime | 支持 Plan / Execute / Reflect、任务产物、失败恢复、安全门和运行指标。 |
| Agentic RAG | 知识写入、切块、检索、来源引用、证据质量门和缺证提示。 |
| Skills / MCP 治理 | 工具 Schema、权限声明、TTL 缓存、熔断器、调用日志和指标诊断。 |
| 分层记忆 | Working / Episodic / Profile Memory，支持多轮审计上下文保留。 |
| Evaluation Harness | Agent / RAG / Research 评测、基线对比、release gate、badcase 沉淀。 |
| 自进化诊断 | 将大厂 Agent 面经/JD 关注的 Runtime、RAG、Tool、Memory、评测和生产化问题转为可执行诊断。 |
| 生产化基础 | FastAPI、Docker、健康检查、CORS、本地持久化、可选 MySQL / Neo4j / 云平台配置。 |

## Web 截图

| 工作台首页 | 审计项目工作台 |
| --- | --- |
| ![工作台首页](docs/screenshots/auditpilot-overview-desktop.png) | ![审计项目工作台](docs/screenshots/auditpilot-audit-workbench.png) |

| Agent 协作 | Agent 运行时 |
| --- | --- |
| ![Agent 协作](docs/screenshots/auditpilot-agent-chat.png) | ![Agent 运行时](docs/screenshots/auditpilot-agent-runtime.png) |

| 移动端首页 |
| --- |
| ![移动端首页](docs/screenshots/auditpilot-overview-mobile.png) |

## 快速启动

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.env.example config.env
python start.py
```

## 安全配置

请不要把任何真实密钥提交到 GitHub。推荐做法：

1. 复制 `config.env.example` 为本地 `config.env`。
2. 在 `config.env` 中填写 `DEEPSEEK_API_KEY`、`OPENAI_API_KEY` 或其他 OpenAI-compatible Key。
3. 生产部署时使用 Railway / Fly.io / Render / Docker Secret 等平台 Secret 管理。
4. `config.env`、`.env*`、运行数据、日志、模型文件和本地数据库已写入 `.gitignore`。

不配置模型密钥时，系统仍可运行审计流程、RAG 检索、控制映射、证据分析、Agent Runtime、评测与页面演示，只是 LLM 增强分析会降级。

## 页面地图

| 路由 | 用途 |
| --- | --- |
| `/` | 产品级总览、核心指标、Self-Evolution Harness、审计项目和交付信号。 |
| `/audit` | 从立项到整改关闭的完整审计项目工作台。 |
| `/chat` | 多 Agent 协作入口，输出上下文、风险、证据缺口和质量门。 |
| `/knowledge` | 知识写入、文档切块、RAG 检索和来源验证。 |
| `/skills` | Agent Runtime、Skills、MCP-style Tool Use、质量诊断和自进化控制面。 |
| `/training` | Agent / RAG / Research 评测、基线回归和发布门禁。 |

## 架构概览

```text
审计项目 / 用户输入
  -> Hybrid Intent Router
  -> Working + Episodic + Profile Memory
  -> Planner / Evidence / Control / Risk / Compliance / Remediation Agents
  -> Agentic RAG + Knowledge Graph + Skills / MCP Tools
  -> Safety Gate + Reflection + Human Review
  -> Audit Repository + Evaluation Baseline + Delivery Package
```

系统边界：

- 模型负责理解、归纳、解释和生成建议；证据缺口、质量门、权限、安全门、交付状态保留可审计的确定性逻辑。
- Agent 不替代审计师最终专业判断；证据不足、高风险或低置信度会进入补证和人工复核。

## 测试

```powershell
.\.venv\Scripts\python.exe -m compileall -q agents services rag web tests scripts
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

当前自动化测试覆盖意图路由、分层记忆、Skill 输入治理与缓存、Agent Runtime 反思、评测基线回归、自进化 Harness、质量诊断和全局搜索。

## 项目结构

```text
agents/                 审计 Agent 主链与控制库
services/               Runtime、Memory、Router、Skills、安全、评测、交付
rag/                    Agentic RAG 与持久化知识库
knowledge_graph/        图谱构建与 Neo4j 可选接入
training/               Agent 评测与离线训练入口
web/                    FastAPI 应用与 API
templates/ + static/    产品界面
tests/                  自动化回归测试
docs/                   项目归档、面试材料与截图
data/                   本地运行数据，大部分已被 Git 忽略
```

## 真实性说明

- 项目中的运行记录、工具调用、评测、记忆和审计档案由真实代码生成并持久化，不是静态截图。
- 未经压测验证的吞吐、准确率或可用性不作为项目事实；简历量化应使用实际评测结果。
- SFT / RLHF 类重训练被明确放在离线任务中；Web 进程提供评测和数据准备入口。

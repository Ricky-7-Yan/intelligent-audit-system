<p align="center">
  <img src="static/assets/auditpilot-logo.svg" width="108" alt="AuditPilot logo" />
</p>

<h1 align="center">审脉 AuditPilot</h1>

<p align="center">
  面向审计交付场景的企业级 AI Agent 工作台：把证据检索、工具执行、Harness 评测、人工复核与整改闭环放进同一条可追溯工作流。
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img alt="FastAPI" src="https://img.shields.io/badge/FastAPI-Agentic%20Backend-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img alt="RAG" src="https://img.shields.io/badge/Agentic_RAG-Evidence_Grounded-0F766E?style=flat-square" />
  <img alt="Evaluation" src="https://img.shields.io/badge/Eval_Harness-Release_Gate-2563EB?style=flat-square" />
  <img alt="Secrets" src="https://img.shields.io/badge/Secrets-Not_Committed-10B981?style=flat-square" />
</p>

![AuditPilot 工作台](docs/screenshots/auditpilot-overview-desktop.png)

## Why AuditPilot

很多 Agent 项目停留在聊天框或 Demo。AuditPilot 选择一个更“硬”的落地场景：企业审计交付。它需要证据、控制、风险、复核、报告和整改闭环，也天然要求可追溯、可回归、可解释。

AuditPilot 的目标不是替代审计师，而是把审计师反复执行的取证、映射、检查、补证和交付动作，组织成一套可治理的 Agent 工作流。

## What it does

| 模块 | 能力 |
| --- | --- |
| Audit Workspace | 审计立项、控制矩阵、审计程序、抽样计划、发现、整改和交付包。 |
| Agent Runtime | 有界 Plan / Execute / Reflect 循环、步骤依赖、运行预算、任务产物、失败恢复和人工复核出口。 |
| Agentic RAG | 知识写入、切块、检索、来源引用、证据质量门和缺证提示。 |
| Skills / MCP-style Tools | 工具 Schema、权限声明、TTL 缓存、熔断器、调用日志和工具指标。 |
| Memory | Working / Episodic / Profile Memory，保留多轮审计上下文。 |
| Evaluation Harness | 对任务结果、执行轨迹、工具调用、证据依据、安全权限、上下文和鲁棒性分层评测；输出校准得分与置信下界，关键断言失败直接阻断发布。 |
| Evidence Graph | 连接任务、步骤、工具运行和产物，检查来源覆盖、断裂依赖与关键孤点。 |
| Governed Improvement | 失败只沉淀为经验候选，通过回归评测和人工批准后才允许复用。 |
| Episode & Observability | 隐私友好的任务轨迹包、标准语义字段、工具证据、安全门、失败归因、干预记录和完整性摘要。 |

## Screenshots

| Audit workspace | Agent runtime |
| --- | --- |
| ![审计项目工作台](docs/screenshots/auditpilot-audit-workbench.png) | ![Agent 运行时](docs/screenshots/auditpilot-agent-runtime.png) |

| Agent collaboration | Layered evaluation |
| --- | --- |
| ![Agent 协作](docs/screenshots/auditpilot-agent-chat.png) | ![分层评测](docs/screenshots/auditpilot-evaluation.png) |

<p align="center">
  <img src="docs/screenshots/auditpilot-overview-mobile.png" width="360" alt="AuditPilot mobile overview" />
</p>

## Architecture

```text
Audit request
  -> Hybrid Intent Router
  -> Working + Episodic + Profile Memory
  -> Planner / Evidence / Control / Risk / Compliance / Remediation Agents
  -> Bounded dependency-aware Agent Loop
  -> Agentic RAG + Evidence Graph + Skills / MCP-style Tools
  -> Safety Gate + Reflection + Human Review
  -> 9-layer Component Evaluation + Release Gate
  -> Delivery Package + Governed Experience Candidate
```

Design boundaries:

- LLMs help with understanding, summarization and explanation.
- Evidence gaps, quality gates, permissions, risk signals and delivery state stay auditable.
- High-risk or low-confidence outputs are routed to evidence completion and human review.

## 快速运行

要求 Python 3.10 或更高版本。项目默认支持无模型密钥运行：未配置大模型时会进入确定性回退模式，审计工作流、RAG、评测和界面仍可体验。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item config.env.example config.env
python start.py
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp config.env.example config.env
python start.py
```

启动完成后，按终端输出的地址在浏览器中打开应用。停止服务时在终端按 `Ctrl+C`。

## 基本操作

1. 在「Agent 协作」中输入审计对象、标准或风险场景，观察多 Agent 的规划、证据、控制、风险与整改协作。
2. 在「审计项目」中建立审计范围，运行控制测试、Deep Research、证据请求和整改闭环。
3. 在「知识库」中维护可检索的审计知识；首次运行会从公开种子知识自动建立本地 RAG 存储。
4. 在「Agent 评测」中运行分层评测，查看任务、轨迹、工具、证据、安全、上下文与鲁棒性结果，以及发布门禁。
5. 在「Skills / MCP」中检查工具 Schema、权限声明、调用记录和运行状态。

## 可选配置

复制 `config.env.example` 后，只在本地 `config.env` 中填写所需配置：

| 场景 | 配置项 | 是否必需 |
| --- | --- | --- |
| LLM 增强 | `DEEPSEEK_API_KEY` 或兼容提供商变量 | 否 |
| MySQL 标准库 | `MYSQL_HOST`、`MYSQL_USER`、`MYSQL_PASSWORD` | 否 |
| Neo4j 图谱 | `NEO4J_URI`、`NEO4J_USER`、`NEO4J_PASSWORD` | 否 |
| RAG 参数 | `RAG_CHUNK_SIZE`、`RAG_CHUNK_OVERLAP`、`RAG_TOP_K` | 否 |

不要在代码、README、Issue 或提交记录中填写真实密钥。

## Security

Do not commit real API keys.

- Put local secrets in `config.env`.
- Use platform secrets for deployment.
- `config.env`, `.env*`, runtime data, logs, model artifacts and local databases are ignored by Git.
- Example config files use placeholders only.

## Public repository scope

公开仓库只保留可运行产品代码、必要的公开种子数据、回归测试、展示截图、README 与启动配置。以下内容不会进入 Git：

- API 密钥、数据库密码和个人配置；
- 本地运行记录、会话记忆、评测结果、上传文件和生成的 RAG 存储；
- 求职 JD、面经、项目内部设计文档和完整归档；
- 本地模型、IDE 配置、日志及临时生成文件。

## Validation

```powershell
.\.venv\Scripts\python.exe -m compileall -q agents services rag web tests scripts
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
```

Current tests cover intent routing, memory compaction, skill validation/cache, bounded runtime execution, component assertions, evidence lineage, governed experience reuse, evaluation regression, repository-bound Harness gating, quality diagnostics and global search.

## Project layout

```text
agents/                 audit agent chain and control library
services/               runtime, memory, router, skills, safety, evaluation, delivery
rag/                    agentic RAG and local knowledge-store runtime
knowledge_graph/        optional graph construction and Neo4j adapter
training/               evaluation and offline training entry points
web/                    FastAPI application and APIs
templates/ + static/    product UI
tests/                  regression tests
docs/screenshots/       public product screenshots only
data/seed_knowledge/    public starter knowledge; runtime data is ignored
```

## Reality notes

- Runtime records, tool calls, evaluations, memory and audit cases are generated by real code and persisted locally.
- Untested throughput, accuracy or availability claims are intentionally not presented as project facts.
- SFT / RLHF-style model training is treated as an offline extension; the Web app focuses on workflow, evaluation and data preparation.

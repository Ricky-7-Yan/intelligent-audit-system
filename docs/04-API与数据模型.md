# API 与数据模型

启动后访问 `/docs` 查看可交互 OpenAPI。以下为主要业务接口。

## 1. Agent 与会话

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/chat` | 融合路由、记忆、审计 Agent 主链 |
| POST | `/api/agent/route` | 只预览意图和 Agent 路由 |
| GET | `/api/session/history/{session_id}` | 内存历史与持久化记忆 |
| GET | `/api/memory/sessions` | 会话列表和记忆统计 |
| GET | `/api/memory/sessions/{session_id}` | 工作/情景/画像完整数据 |

`POST /api/chat`：

```json
{
  "message": "分析 ERP 权限日志证据并生成整改计划",
  "session_id": "demo-001",
  "context": {"audit_period": "2026 Q2"}
}
```

响应包含 `routing`、`memory`、`audit_context`、`task_plan`、`retrieval`、`evidence_pack`、`control_matrix`、`risk_assessment`、`quality_gate`、`recommendations` 和 `execution_trace`。

## 2. 审计项目

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/audit` | 创建并执行审计项目 |
| GET | `/api/audit/templates` | 场景模板 |
| GET | `/api/audit/controls` | 控制库 |
| GET | `/api/audit/runs` | 审计档案 |
| GET | `/api/audit/runs/{run_id}` | 完整档案 |
| POST | `/api/audit/runs/{run_id}/review` | 人工复核 |
| POST | `/api/audit/runs/{run_id}/tasks/{task_id}` | 整改任务更新 |
| POST | `/api/audit/runs/{run_id}/evidence/{request_id}` | 证据请求更新 |
| POST | `/api/audit/runs/{run_id}/controls/{control_id}/test` | 控制测试更新 |
| GET | `/api/audit/runs/{run_id}/delivery` | 交付包 JSON |
| GET | `/api/audit/runs/{run_id}/delivery.md` | 交付包 Markdown |

审计档案核心字段：

```text
request -> result -> evidence_requests -> control_tests
        -> remediation_tasks -> reviews -> event_log
        -> evidence_analyses -> lifecycle_stage
```

## 3. 证据与知识

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/evidence/analyze` | 上传并分析证据文件 |
| GET | `/api/evidence/analyses` | 分析历史 |
| POST | `/api/knowledge/add` | 写入文本知识 |
| POST | `/api/knowledge/upload` | 上传并切块 |
| GET | `/api/knowledge/query` | RAG 查询 |
| GET | `/api/knowledge/stats` | 知识统计 |
| POST | `/api/knowledge/build` | 构建实体关系图 |

## 4. Runtime、Skills 与安全

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/agent/tasks` | 创建协议化任务并执行首步 |
| GET | `/api/agent/tasks` | 任务列表 |
| POST | `/api/agent/tasks/{task_id}/run-next` | 执行下一步 |
| POST | `/api/agent/tasks/{task_id}/steps` | 添加人工步骤 |
| GET | `/api/agent/observability` | 任务、反思、重试、延迟、工具指标 |
| GET | `/api/skills` | Skill 元数据和可靠性配置 |
| GET | `/api/mcp/tools` | MCP 风格工具描述 |
| POST | `/api/skills/{skill_name}/run` | 执行 Skill |
| GET | `/api/skills/runs` | 调用日志 |
| POST | `/api/safety/check` | 安全门禁 |

## 5. 评测

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| POST | `/api/training/evaluate` | Agent 端到端评测 |
| POST | `/api/evaluation/rag` | RAG 评测 |
| GET | `/api/evaluation/runs` | 评测历史、基线差异、发布门禁 |
| POST | `/api/research/answer` | Deep Research |
| GET | `/api/research/evaluation-plan` | Research 评测计划 |

## 6. 错误约定

- `400`：文件类型、大小或 JSON 上下文错误。
- `404`：任务、档案、证据分析或 Skill 不存在。
- `422`：Pydantic 请求字段校验失败。
- Skill 内部失败通过运行记录返回 `status=failed`、`error_type`、`circuit_state`。

## 7. 持久化目录

| 目录 | 内容 |
| --- | --- |
| `data/audit_runs/` | 审计项目与闭环状态 |
| `data/agent_runtime/` | Runtime 任务检查点 |
| `data/conversation_memory/` | 三层会话记忆 |
| `data/evaluation_runs/` | 评测、基线与发布门禁 |
| `data/evidence_analyses/` | 证据文件分析 |
| `data/skill_runs/` | Skill 调用 JSONL |
| `data/rag_store/` | RAG 文档切片 |

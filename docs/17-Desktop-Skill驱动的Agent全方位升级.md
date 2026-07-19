# Desktop Skill 驱动的 Agent 全方位升级

## 1. 升级目标

本轮依据桌面 `skill` 资料库中与 AuditPilot 直接相关的 Agent 工程、评测、记忆、工具、RAG、前端和验证技能，对项目进行一次系统升级。目标不是增加展示名词，而是把 Harness、自进化、上下文治理和多 Agent 可观测性落实为可运行代码、持久化数据、API、界面和测试。

## 2. 使用的 Skill 与落地映射

| Skill | 核心原则 | 本项目落地 |
|---|---|---|
| `harness-engineering` | 锁定面、可编辑面、追加日志、人工控制面分离 | `HarnessControlPlane`、锁定文件指纹、事件日志、候选档案、人工审批 |
| `self-improvement-loops` | 双数据集无回归、严格提升、拒绝档案、外部评测器 | Held-in / Held-out 验收门、拒绝记录、禁止自动推广 |
| `context-optimization` | JIT 装载、预算、遮罩、压缩、稳定前缀 | Memory 上下文预算、估算利用率、按需历史、遮罩引用、稳定分区 |
| `memory-systems` | 分层记忆、时效、隐私、删除权、渐进增强 | Working / Episode / Profile、validity、privacy、记忆删除 API |
| `tool-design` | 无歧义契约、可恢复错误、命名空间、日志脱敏 | MCP qualifiedName、outputSchema、错误码/恢复建议、输入脱敏 |
| `multi-agent-patterns` | 角色隔离、显式交接、验证后传递 | role-level trace、depends_on、handoff、产物引用、预算硬门 |
| `eval-harness-first` | 确定性优先、固定门禁、基线与回归 | 本地确定性评测、持久化评测历史、发布门禁、预热 |
| `rag-architect` | 检索评测、来源与质量指标、生产级回归 | RAG 评测继续保留多用例并行，启动时预热首个 RAG case |
| `impeccable` / `responsive-design` / `interaction-design` | 产品型界面、状态动效、触控、响应式、减少认知负荷 | 专注视图、折叠分区、候选档案、移动端触控、无横向溢出、reduced-motion |

## 3. Harness 与自进化

新增 `services/harness_control.py`：

- 将评测器、训练评测逻辑和核心测试声明为锁定面，并保存 SHA-256 指纹。
- 每个候选只允许选择一个已声明的可编辑表面，避免一次改动跨越过多边界。
- 候选依次经历 `draft → awaiting_human_review → approved/rejected → archived`。
- 验收要求 Held-in 与 Held-out 均不回归，且至少一项指标严格提升。
- 确定性检查必须全部通过；锁定面变化会直接拒绝候选。
- 自动门禁通过后仍不会自动上线，推广必须由人工审批。
- `events.jsonl` 与 `rejected.jsonl` 为追加式记录；界面的“归档”可恢复，不破坏审计链。

新增 API：

- `GET /api/agent/harness`
- `POST /api/agent/harness/candidates/{candidate_id}/evaluate`
- `POST /api/agent/harness/candidates/{candidate_id}/review`
- `DELETE /api/agent/harness/candidates/{candidate_id}`

## 4. Runtime、多 Agent 与 Tool Use

AgentRuntime 新增：

- 每步角色级 Trace：角色、输入哈希、Skill、决策、依赖、交接、耗时、尝试次数和产物引用。
- 工具预算硬限制；预算耗尽时进入人工复核，不允许继续隐式调用。
- 任务创建、步骤完成、预算耗尽写入追加事件日志。

SkillRegistry 新增：

- `AuditPilot:<tool_name>` 完整命名空间。
- 统一输出 Schema。
- `INPUT_VALIDATION_ERROR`、`CIRCUIT_OPEN`、`TOOL_TIMEOUT`、`TOOL_EXECUTION_ERROR` 等结构化错误。
- 每个错误提供 `retryable` 和具体恢复动作。
- API Key、Token、Secret、Password、Authorization 和 Cookie 自动脱敏后再写日志。

## 5. Memory 与上下文效率

ConversationMemory 新增：

- 显式上下文预算和 token 估算。
- 只装载摘要、画像、相关历史和最近对话，避免全量记忆塞入 Prompt。
- 超预算内容使用可追溯遮罩提示，不修改持久化原文。
- Episode 增加 `confidence`、`status`、`valid_from`、`valid_until`、`privacy` 和反馈计数。
- 新增会话删除能力与 `DELETE /api/memory/sessions/{session_id}`，满足记录管理和隐私删除权。

## 6. 评测性能

首轮基准发现：共享生产 MySQL 连接进行并发评测会导致连接协议错误和明显等待。最终方案为：

- 评测 Agent 显式禁用外部 MySQL / Neo4j 连接。
- 使用内置审计标准和确定性本地逻辑，保持同一评分输出。
- 对轻量 Agent case 使用顺序执行，因为实际测量显示线程调度成本高于单 case 计算成本。
- RAG case 仍保留并行检索评测，并在应用启动阶段预热一个代表性 case。
- API 层继续使用结果缓存，重复评测可直接返回缓存结果。

这体现了 Harness 的核心原则：先测量，再选择并行、缓存或隔离；不能为了“看起来先进”引入更慢且不安全的并发。

## 7. 界面与交互

Agent 运行时中心新增 Harness 候选档案，并进一步降低长页面负担：

- 页面导航保留快速定位，并新增“专注视图”：仅展示当前分区，一键恢复全部。
- 治理细节、质量诊断、候选档案继续使用渐进展开，默认不向客户堆叠全部技术信息。
- 候选支持归档管理，事件链仍然保留。
- 质量诊断直接读取 Harness 控制面，展示锁定评测表面、双数据集门禁与人工审批证据，不再把已落地能力误报为缺失。
- 修复控制平面角色和描述文字贴合问题。
- 标题、长文本和记录 ID 支持安全换行，避免窄屏重叠。
- 移动端关键按钮采用更大的触控区域，无横向溢出。
- 动效仅用于状态变化和展开反馈，并继续尊重 `prefers-reduced-motion`。
- 市场趋势和岗位研究继续放在项目文档中，不新增到客户产品页面。

## 8. 验证口径

完成声明必须同时满足：

1. Python 编译检查通过。
2. JavaScript 语法检查通过。
3. 单元测试覆盖 Harness 双集门禁、人工审批、归档、Memory 预算和删除权。
4. 浏览器验证 `/skills` 页面身份、非空渲染、控制台健康、专注视图交互、Harness 展开、桌面与移动端无横向溢出。
5. Git 保存本轮变更，保留可追溯版本。

本轮实测：

- 9 个单元测试全部通过，总用时 0.135 秒。
- Agent 评测 API 首次请求约 387ms、缓存请求约 12ms；RAG 评测首次约 16ms、缓存请求约 12ms。
- 390px 移动端无横向溢出，所有可见操作控件达到 44px；Harness lane 标题与说明无文字重叠。

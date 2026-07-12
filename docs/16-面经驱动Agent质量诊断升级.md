# 16 面经驱动 Agent 质量诊断升级

本文记录本轮升级：把大厂 Agent 面经中面试官最在意的追问点，落到项目的真实代码、接口、页面和测试里，而不是只停留在简历描述。

## 01 升级目标

面经高频问题通常集中在六类：

1. Agent 是否只是 Chatbot，还是有规划、执行、反思、恢复和可观测运行时。
2. RAG 如何处理召回、错召、漏召、证据不足和幻觉。
3. Tool Use / MCP 是否有 Schema、权限、缓存、熔断、日志和失败恢复。
4. Agent/RAG 效果如何评测，badcase 如何沉淀，如何防止改坏。
5. Memory 如何分层、压缩、召回、删除，如何避免上下文污染。
6. 项目是否真实生产化，而不是 Demo；包括日志、质量门、降级、迁移路径和测试。

本轮将这些问题做成可运行的质量诊断服务和 `/skills` 页面诊断面板。

## 02 新增代码能力

### 2.1 AgentQualityDiagnostics

新增文件：

- `services/agent_quality.py`

新增服务会聚合真实运行状态：

- `AgentRuntime.observability()`：任务数、工具调用、反思记录、延迟、安全门。
- `SkillRegistry.metrics()`：工具运行次数、成功率、缓存命中、熔断器状态。
- `ConversationMemory.stats()`：会话、轮次、episode、工作记忆。
- `EvaluationRunRepository.list_runs()`：Agent/RAG/Research 评测历史、release gate、blockers。
- RAG statistics：知识文档与切片数量。

输出内容包括：

- 总分和面试可讲状态。
- 六个维度评分：Agent Runtime、RAG Grounding、Tool/MCP、Evaluation Harness、Memory、Production Engineering。
- badcase 诊断：release gate blocker、工具失败、安全门阻断。
- tool use 诊断：工具选择契约、Top tools、缓存、熔断器。
- RAG debug playbook：从 golden set、chunk、top-k、rerank 到 missing evidence 的排查路径。
- production readiness：可观测、日志、RAG、Memory、安全门、评测门禁等检查。
- interview pitch：用于面试现场解释“为什么这么设计”的简短话术。

### 2.2 API 接口

新增接口：

```http
GET /api/agent/quality-diagnostics
```

该接口不是静态文案，而是实时读取项目当前运行数据，适合演示给面试官看：

- 当前哪些能力已经有证据。
- 哪些能力仍是 gap。
- gap 应该如何补。
- 为什么当前架构可迁移到 LangGraph、Temporal、真实 MCP Server、数据库、队列和 Redis。

### 2.3 Tool Registry 指标补齐

更新文件：

- `services/skill_registry.py`

`metrics()` 新增：

- `skills`：当前注册工具数量。
- `open_circuits`：当前打开的熔断器数量。
- `circuits`：每个工具熔断器状态、失败次数和重试剩余时间。

这样 Harness、自进化和质量诊断都能共用同一套真实工具治理指标。

## 03 页面升级

更新文件：

- `templates/skills.html`
- `static/skills.js`
- `static/app.css`

`/skills` 页面新增“面经驱动质量诊断”折叠面板：

- 顶部显示总分、能力维度、Badcase 信号、生产化检查通过率。
- 面试讲法以简洁卡片展示，避免页面过长。
- 六个能力维度使用可展开卡片展示：证据、gap、下一步动作。
- Badcase、Tool Use、Production 三组细节分栏展示。
- 加入轻量动态光晕、悬浮反馈和折叠交互，提升质感但不增加操作负担。

这也回应了用户之前提出的要求：页面不要太长、记录可折叠、信息可管理、功能不减少。

## 04 为什么这样设计

### 4.1 为什么不用纯文档说明

面试官通常会追问“你这个项目是真的跑起来了吗”。因此本轮把面试话术绑定到真实数据：

- 运行任务证明 Agent Runtime 不是 Chatbot。
- 工具日志证明 Tool Use 可观测。
- 评测仓库证明效果可回归。
- release gate blocker 证明坏例不会被忽略。
- RAG stats 证明知识库状态可检查。
- Memory stats 证明上下文不是一次性 prompt 拼接。

### 4.2 为什么不用一个大分数

Agent 项目失败通常不是单点问题，而是检索、工具、记忆、评测、运行时和生产工程共同决定。因此采用六维评分：

- 面试时能分维度解释。
- 开发时能定位下一步优先级。
- 后续可以把每个维度扩展为独立 benchmark。

### 4.3 为什么 badcase 要单独做

大厂面试非常关注 badcase 处理。项目现在会把以下问题沉淀为 badcase signal：

- 评测 release gate 未通过。
- 工具失败。
- 安全门阻断。

这样可以讲清楚闭环：发现问题 → 分类 → 修复 → 回归评测 → 再发布。

### 4.4 为什么 Tool Use 要有缓存和熔断

Tool Use 的真实问题不是“能不能调用函数”，而是：

- 参数是否符合 schema。
- 权限是否被约束。
- 高频调用是否浪费成本。
- 外部工具失败是否拖垮 Agent。
- 调用失败是否可追踪。

因此项目把工具抽象为 `SkillRegistry`，并提供 schema、permission、TTL cache、run log、circuit breaker、MCP-style 描述。

## 05 面试可讲法

可以这样介绍本轮能力：

> 我没有把面经问题只写进文档，而是做成了一个可运行的 Agent 质量诊断服务。它会读取 Runtime、Tool、Memory、Evaluation、RAG 的真实状态，按 Agent Runtime、RAG、Tool/MCP、Harness、Memory、生产工程六个维度评分。面试官问到 badcase、工具失败、RAG 幻觉、Memory 压缩、release gate 时，我可以直接打开 `/skills` 页面展示证据、gap 和下一步动作。

如果被问“为什么不用 LangGraph / Temporal / 真 MCP”，可以回答：

> 当前项目用轻量自研 Runtime 是为了让项目在本地可完整演示，并把审计业务链路、证据质量门、工具日志和评测闭环先打通。生产环境迁移时，Runtime 的 plan/execute/reflect 可以映射到 LangGraph 状态图或 Temporal workflow；SkillRegistry 的 schema/permission/log 可以迁移到真实 MCP server；JSON 持久化可以替换成数据库、对象存储和队列。

## 06 验证

本轮新增并通过：

- `node --check static\skills.js`
- `python -m compileall -q services web tests`
- `python -m unittest discover -s tests -p "test_*.py"`

新增测试覆盖：

- `SkillRegistry.metrics()` 暴露 `skills`、`open_circuits`、`circuits`。
- `AgentQualityDiagnostics` 能从 Runtime、RAG eval、Tool log、Memory state 生成六维质量报告。

## 07 后续可继续增强

1. 把每个质量维度拆为独立 benchmark，并保存趋势。
2. 给 badcase 增加根因分类：planning、retrieval、tool_schema、memory_conflict、evidence_missing。
3. 接入真实 MCP server 后，对工具选择准确率、工具参数错误率做专项评测。
4. 把 production readiness 接到部署检查：数据库、队列、缓存、对象存储、日志检索、权限审计。
5. 增加页面上的“生成面试讲解稿”按钮，将当前诊断结果转为 STAR 项目讲述。

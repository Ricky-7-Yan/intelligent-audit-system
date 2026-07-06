# Agent 核心能力详解

## 1. 融合意图路由

`HybridIntentRouter` 不依赖外部模型，融合：

- Pattern：范围、证据、控制、风险、合规、整改、知识等领域关键词。
- Local semantic：字符 2/3-gram 和词项向量余弦相似度。
- Entity：标准、系统、控制编号。
- Urgency：重大、泄露、舞弊、监管、紧急等表达。

输出保留各分支分数，便于复盘错误路由。高分候选会形成多 Agent 协作，而不是强制只选一个 Agent。

## 2. 分层记忆

`ConversationMemory` 提供：

- Working Memory：当前会话最近消息。
- Episodic Memory：超过阈值后压缩旧消息并形成 Episode。
- Profile Memory：标准、风险主题、系统、最近意图和 Agent。
- Related Memory：按词项 Jaccard 相似度召回相关历史。

数据按会话写入 `data/conversation_memory/`，采用临时文件替换以降低半写入风险。该实现可直接运行；生产环境可将同一接口迁移到 Redis + 向量数据库。

## 3. Plan / Execute / Reflect

Runtime 将任务分为：

1. Planning Agent：审计范围与交付物。
2. Control Agent：风险到控制矩阵。
3. Evidence Agent：证据请求、质量规则和采集方法。
4. Remediation Agent：责任人、期限、状态和关闭标准。

每个步骤记录依赖关系、输入、运行、重试、工具状态和产物。执行后形成 Reflection：

- `pass`：产物充分，进入下一步。
- `review`：输出过短或不确定，需要复核。
- `human_review`：工具失败或阻断，需要人工决定重试或改写计划。

## 4. Skills / MCP 工具治理

每个 Skill 包含：

- 名称、标题、描述和版本。
- JSON Schema 风格输入定义。
- 权限声明。
- 超时、缓存 TTL 和失败熔断阈值。
- Handler 与可观测运行日志。

执行顺序：

```text
Schema 校验 -> 熔断检查 -> TTL 缓存 -> 超时执行 -> 统计与日志 -> 反思
```

知识检索 Skill 默认缓存 120 秒；连续失败达到阈值后熔断 30 秒。

## 5. Agentic RAG

RAG 不只是 `/search` 旁路，而是审计主链的一部分：

- 文档切块与 metadata。
- 本地持久化。
- 语义/关键词混合召回。
- 查询扩展和多路结果合并。
- 来源、相关度和置信度。
- 向量服务不可用时的确定性降级。

审计 Agent 会同时融合 RAG、知识图谱、标准库和控制库，质量门再检查来源与证据充分性。

## 6. 风险与质量门

风险评分考虑：

- 领域风险词与固有风险。
- 控制成熟度与控制抵减。
- RAG/证据数量和质量。
- 缺失的设计证据与运行证据。

质量门输出：

- `confidence`、`evidence_grounding`。
- `missing_evidence`。
- `escalation_required`。
- 自动继续、补证或人工复核状态。

模型无法绕过质量门直接把推测写成最终审计结论。

## 7. 评测与自进化闭环

评测维度包括：

- Faithfulness、Completeness。
- Audit Professionalism、Actionability。
- Compliance Alignment。
- Agentic Capability、Tool Trace Quality。
- Human Review Awareness。

每次评测持久化并和同类型上一基线比较：

- 整体分或通过率下降超过 5%：标记回归。
- 平均延迟上升超过 25%：标记性能回归。
- 整体分、通过率和回归数共同决定发布门禁。

Badcase 可转成自定义用例和知识/Skill 更新，形成“发现问题 -> 修正 -> 回归 -> 发布”的闭环。

## 8. 与 EchoMind 的继承与升级

吸收：

- 融合意图识别、分层记忆、多 Agent 协作。
- 查询改写、RAG、Skills、工具可靠性。
- Monitor、LLM-as-Judge 和回归评测思想。

保留并加强：

- 审计控制库、证据文件分析、控制测试、底稿与交付包。
- 风险质量门和人工复核。
- 审计发现、整改和关闭状态。

因此项目不是 EchoMind 的客服换皮，而是同类 Agent 工程能力在审计行业中的完整产品化。

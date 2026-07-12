# 15 大厂 Agent 面经驱动面试指导

> 本文基于 2026-07-12 检索整理的 52 份公开面经/面经合集，以及 AuditPilot 当前项目实现编写。  
> 目标：面试时不只“背项目”，而是能回答设计、实现、取舍、缺陷、演进和业务价值。

## 1. 面试定位：这个项目应该怎么讲

一句话：

> 审脉 AuditPilot 是一个面向 IT 审计、内控和合规交付的企业级 AI Agent 工作台，把审计项目从立项、取证、控制测试、风险识别、整改、人审到交付包下载串成一条可追踪链路，并在链路中落地 Router、Memory、Multi-Agent、RAG、Tool Use、评测回归、发布门禁和自进化 Harness。

不要把它讲成“我做了一个聊天机器人”。更好的定位是：

- 业务对象不是聊天消息，而是审计项目。
- Agent 不只生成文本，还要产出控制矩阵、证据请求、审计发现、整改计划、报告和交付包。
- 系统不是只追求回答漂亮，而是强调可追溯、可复核、可评测、可治理。
- 对齐大厂 Agent JD：Planning、Memory、Tool Use、RAG、Evaluation、Harness、Human-in-the-loop、工程化落地。

## 2. 推荐 90 秒项目介绍

我做的是一个审计交付 Agent，而不是单轮问答 Demo。一次请求会先经过 Hybrid Intent Router，输出意图、置信度、实体和候选 Agent；然后读取工作记忆、情景记忆和画像记忆，把上下文传给 Audit Agent。核心执行层是自研的 Plan/Execute/Reflect Runtime，会把任务拆成 Planning、Control、Evidence、Risk、Compliance、Remediation 等角色步骤，每一步可以调用受治理的 Skills/MCP-style 工具，比如 RAG 查询、控制映射、证据检查、报告打包等。

RAG 不是旁路搜索，而是接在审计主链里：文档会切块持久化，查询会改写成多路检索，召回结果带来源、分数和置信度；如果证据不足，质量门不会让模型强行下结论，而是生成补证或人工复核任务。系统还内置 Agent/RAG/Deep Research 评测，记录历史基线、质量回归和延迟回归，最后由发布门禁决定是否可交付。最近我又补了 Self-Evolution Harness，把评测退化、工具失败、记忆状态和 JD 能力缺口聚合成优化提案，并能一键物化成 Runtime 任务。

## 3. 面经高频问题与项目回答

### 3.1 你这个项目和普通 ChatGPT 套壳有什么区别？

普通套壳通常是“用户输入 → Prompt → LLM 输出”。AuditPilot 做的是审计业务闭环和 Agent 工程链：

1. 有明确业务对象：审计项目、证据、控制测试、整改任务、复核记录、交付包。
2. 有执行过程：Router、Memory、任务计划、工具调用、反思、质量门。
3. 有工具治理：Schema 校验、权限、缓存、熔断、调用日志。
4. 有评测闭环：Agent/RAG/Research 评测、历史基线、回归检测、发布门禁。
5. 有人工复核：证据不足、高风险、低置信度不会自动给强结论。

可以补一句：

> 我把 LLM 放在可治理流程里，而不是把业务风险交给一段不可解释的自然语言输出。

### 3.2 Agent 总体架构怎么设计？

推荐按链路讲：

```text
用户请求
  -> Hybrid Intent Router
  -> Conversation Memory
  -> Audit Agent / Research Agent
  -> Plan/Execute/Reflect Runtime
  -> Skills / MCP-style Tools
  -> Agentic RAG / 控制库 / 证据分析
  -> Quality Gate / Safety Gate
  -> Evaluation Repository / Release Gate
  -> 审计报告与交付包
```

实现锚点：

- `services/intent_router.py`：Pattern + 本地 n-gram 相似度融合路由。
- `services/conversation_memory.py`：工作记忆、情景记忆、画像记忆。
- `services/agent_runtime.py`：任务协议、步骤、依赖、反思、产物、指标。
- `services/skill_registry.py`：工具注册、Schema、缓存、熔断、MCP-style 描述。
- `rag/agentic_rag.py`：切块、持久化、混合检索、查询扩展、来源置信度。
- `training/training_pipeline.py`、`services/rag_evaluator.py`：评测。

### 3.3 为什么要做 Hybrid Intent Router，不直接让 LLM 判断意图？

面试官关注的是稳定性、成本和可解释性。

我没有完全依赖 LLM 做路由，因为企业审计场景里，路由错误会导致调用错工具、产出错交付物。Hybrid Router 用规则信号兜底审计领域关键词，再用本地 n-gram 相似度增强泛化，输出每个候选意图的分数、实体、紧急度和协作策略。

优点：

- 无模型密钥也能运行，便于演示和降级。
- 延迟低、成本低，路由过程可解释。
- 可以看见候选 Agent 分数，便于排查错误路由。

为什么不用纯规则：

- 纯规则泛化差，同义表达容易漏。
- 本地相似度可以覆盖“权限滥用/越权访问/职责分离不足”等不同说法。

为什么不用纯向量：

- 当前审计意图集合较小，纯向量服务会增加部署复杂度。
- 生产可把相同接口替换为 embedding/分类模型，但项目阶段先保证可运行、可解释、可降级。

### 3.4 Multi-Agent 怎么避免“为了多而多”？

我没有把所有问题都拆成多个 Agent。Router 只有在多个候选意图分数同时达到协作阈值时才触发协作。每个 Agent 不是聊天角色，而是绑定不同输入、工具和产物：

- Planning：审计范围、目标、交付物。
- Control：风险到控制矩阵。
- Evidence：证据请求、证据质量规则。
- Risk：风险评分和残余风险。
- Compliance：标准条款和合规映射。
- Remediation：整改计划、责任人、关闭条件。
- Quality Gate：证据充分性、置信度和人审判断。

最后由质量门统一收敛，避免多个 Agent 输出互相冲突。

### 3.5 为什么自研 Runtime，而不是直接 LangGraph？

我保留了自研轻量 Runtime，主要是为了让作品集和面试表达更透明：任务协议、步骤依赖、重试、反思、工具状态、产物和指标都能直接解释，也能贴合审计业务的交付对象。LangGraph 很适合复杂状态图和生产级编排，但如果一开始直接套框架，面试时容易只讲“我用了某框架”，讲不清状态协议和失败处理。

取舍：

- 当前：自研 Runtime，低依赖、可解释、便于演示。
- 生产扩展：如果任务规模变大，可把节点迁移到 LangGraph/Temporal/Celery；业务 Agent、Repository、Skill 接口不需要推翻重写。

### 3.6 RAG 是怎么做的？

回答按 pipeline：

1. 文档写入：文本或文件进入 `DocumentProcessor`。
2. 切块：按段落优先，长段落再窗口切分，保留 metadata。
3. 持久化：写入本地 JSON store，便于无数据库环境运行。
4. 查询扩展：根据审计上下文生成多路 query。
5. 混合检索：语义分数、TF-IDF、关键词信号融合。
6. 去重与合并：多路查询结果合并，保留来源和分数。
7. 答案生成：优先基于来源句子生成，给出置信度。
8. 降级：没有 embedding 或 LLM 时仍可用关键词/TF-IDF 跑通。

为什么不用纯向量库：

- 审计知识包含标准条款、控制编号、证据名称，这类 token/编号对关键词很敏感。
- 纯向量容易召回语义相似但标准编号不匹配的内容。
- 混合检索更适合“自然语言问题 + 结构化标准/控制编号”的场景。

生产演进：

- 向量库可换 Milvus/pgvector/Elasticsearch/OpenSearch。
- 重排可接 Cross Encoder 或 LLM reranker。
- 来源可加入权限过滤、版本过滤和权威性分。

### 3.7 如何控制幻觉？

我不是让模型自由生成审计结论，而是用多层约束：

- RAG 返回来源、分数和置信度。
- 审计主链融合标准库、控制库、证据分析和业务上下文。
- Quality Gate 检查证据数量、证据质量、控制覆盖和置信度。
- 证据不足时输出 missing evidence 和补证请求。
- 高风险或低置信度进入 Human-in-the-loop。
- 评测阶段检查 faithfulness、引用覆盖、失败模式。

一句话：

> 系统可以说“目前证据不足，需要补证”，但不能把推测包装成最终审计结论。

### 3.8 Tool Use 怎么设计？

`SkillRegistry` 把工具当成可治理资源，而不是随便让 Agent 调函数。每个 Skill 有：

- 名称、描述、版本。
- JSON Schema 风格输入定义。
- 权限声明。
- 超时配置。
- TTL 缓存。
- 失败阈值和熔断。
- 运行日志和指标。
- MCP-style tools 描述接口。

执行链路：

```text
Schema 校验 -> 熔断检查 -> 缓存检查 -> Handler 执行 -> 日志记录 -> 指标聚合
```

这样回答能覆盖大厂常问的 Function Calling、MCP、工具权限、工具可靠性和可观测性。

### 3.9 为什么要缓存和熔断？

Agent 系统里工具调用会成为可靠性瓶颈。比如 RAG 查询、Deep Research brief、报告打包等工具，如果重复调用会增加延迟和成本；如果某个工具连续失败，Agent 还继续调用，会放大故障。

所以我做了：

- TTL 缓存：相同输入在有效期内直接返回，降低延迟。
- 熔断：连续失败达到阈值后短时间打开 circuit，避免雪崩。
- 日志：记录 cache hit、latency、error type，给评测和 Harness 使用。

### 3.10 评测体系怎么做？

评测分三类：

- Agent 评测：看回答质量、审计专业性、行动建议、工具轨迹、人审意识。
- RAG 评测：看相关性、来源覆盖、忠实度、权威性、失败模式。
- Deep Research 评测：看多路查询、来源融合、推理步骤、证据充分性。

每次评测会写入 `EvaluationRunRepository`，并和同类历史基线对比：

- 分数下降超过阈值：质量回归。
- 延迟上升超过阈值：性能回归。
- 通过率下降或 blocker 增加：发布门禁阻断。

这样可以回答“你的效果怎么证明”“怎么防止改坏”。

### 3.11 Self-Evolution Harness 是什么？怎么不是噱头？

我把自进化限定在可验证的工程闭环里，不让系统自动改代码、自动上线。Harness 会聚合：

- 评测基线和回归。
- Runtime 反思和失败步骤。
- Skill 失败、缓存、熔断和延迟。
- Memory 状态。
- JD 能力项缺口。

然后生成优化提案，提案可以物化成 AgentRuntime 任务，再由人工执行、验证、回归和提交。

边界：

- 当前是“建议和任务化”，不是无人值守自改生产系统。
- 必须经过评测和发布门禁。
- 高风险改动需要人工复核。

这比空喊 self-evolving 更可信。

### 3.12 为什么前端不用 React/Vue？

项目当前是作品集和可运行原型，前端采用模板 + 原生 JS，可以降低部署和构建复杂度，确保在本地、无构建环境和求职演示里快速启动。核心业务逻辑都在 API 和服务层，前端只是展示和交互。

如果生产化：

- 可以把前端迁移到 React/Vue/Next.js。
- API 契约保持不变。
- 命令中心、折叠记录、运行时详情、评测看板都可以组件化。

### 3.13 为什么数据层先用文件 Repository？

文件 Repository 的目的不是替代生产数据库，而是保证项目完整可运行、可演示、可提交 Git，同时让数据结构透明。所有审计运行、评测记录、证据分析、记忆、工具日志都是真实持久化，不是静态 mock。

生产迁移方案：

- 审计项目、任务、评测记录迁移到 PostgreSQL/MySQL。
- 大文件和交付包进入对象存储。
- 运行日志进入 ClickHouse/Elasticsearch/OpenSearch。
- 异步任务进入 Celery/RQ/Kafka/Temporal。

面试时要主动说边界，反而更可信。

### 3.14 评测速度慢怎么优化？

已做：

- 评测 runtime 预热。
- RAG 评测并行执行。
- 相同请求缓存。
- 前端 busy 状态和局部刷新，减少用户等待焦虑。
- Deep link 和全局命令中心减少页面查找时间。

可继续做：

- 增量评测：只跑受影响 case。
- 分层评测：PR 阶段跑 smoke set，发布前跑 full set。
- 模型调用批处理和限流。
- 评测结果按 case hash 缓存。
- 异步任务队列和后台通知。

### 3.15 如果面试官问“为什么不用某某技术？”

统一回答框架：

1. 当前目标是什么：作品集可运行、可解释、可验证。
2. 该技术带来什么收益：性能、规模、生态或生产能力。
3. 为什么当前阶段没引入：复杂度、部署依赖、可解释性或成本。
4. 后续如何平滑迁移：接口隔离、Repository 抽象、Runtime 协议、API 契约。

| 被问技术 | 回答角度 |
| --- | --- |
| LangGraph | 当前自研 Runtime 便于解释状态协议；生产复杂状态图可迁移 |
| Milvus/pgvector | 当前混合检索本地可运行；生产大规模知识库可替换 |
| Kafka/Celery | 当前任务量本地执行足够；生产长任务和异步评测可接 |
| Redis | 当前内存/文件缓存便于演示；生产缓存、锁和 session 可用 Redis |
| Elasticsearch | 当前关键词/TF-IDF 够用；生产日志检索和大规模 BM25 可接 ES/OpenSearch |
| React/Next.js | 当前模板降低部署成本；生产产品化可组件化 |

## 4. 技术拷打专项回答

### 4.1 一次请求从输入到输出发生了什么？

1. 前端页面发请求到 FastAPI。
2. API 调用 SafetyGate 做输入和风险检查。
3. IntentRouter 输出意图、置信度、实体、候选 Agent。
4. ConversationMemory 读取相关上下文。
5. AuditAgent 抽取审计对象、标准、领域和风险主题。
6. RAG 查询标准/知识/控制证据。
7. AuditAgent 生成任务计划、控制矩阵、证据包、风险评分、合规检查。
8. QualityGate 判断证据充分性和是否需要人审。
9. 结果写入 Repository。
10. 前端展示报告、整改任务、证据请求、控制测试、下载入口。

### 4.2 如何设计质量门？

质量门不是一个单独分数，而是多维判断：

- evidence_quality：证据数量、来源、文件画像。
- control_coverage：控制矩阵覆盖度。
- confidence：RAG 置信度、规则信号、证据支撑。
- risk_level：高风险自动提高复核要求。
- missing_evidence：缺失关键证据时不能直接交付。
- escalation_required：是否进入人工复核。

设计原因：

- 审计场景对“证据链”要求高。
- 单纯 LLM 自评不可靠。
- 多维门禁比一个总分更容易解释。

### 4.3 如何设计发布门禁？

发布门禁看历史对比，而不是只看当前分数：

- 当前整体分。
- 当前通过率。
- 与上一个同类基线相比的分数变化。
- 延迟变化。
- 回归数量。
- blocker 风险。

为什么要看基线：

Agent 项目经常“新增能力导致旧能力退化”。基线对比能发现质量回归，而不是每次只看单点样例。

### 4.4 如何处理工具失败？

按严重程度：

- 输入不合法：Schema 直接拒绝，返回可解释错误。
- 暂时失败：允许重试。
- 连续失败：熔断并记录。
- 关键工具失败：Runtime reflection 标记 review/human_review。
- 业务风险高：人工复核。

### 4.5 如何处理长上下文和记忆污染？

当前实现：

- Working Memory 保留近期消息。
- Episodic Memory 对旧消息摘要。
- Profile Memory 存结构化画像。
- Related Memory 用相似度召回相关历史。

为什么分层：

- 近期对话需要顺序。
- 历史事件需要压缩。
- 画像需要结构化更新。
- 相关召回需要按问题检索。

可演进：

- 记忆过期策略。
- 冲突检测。
- 用户可删除记忆。
- 敏感信息脱敏。
- 向量召回 + 结构化过滤。

## 5. 面试官可能继续追问的缺点

### 5.1 当前项目最大不足是什么？

当前最大不足不是功能不完整，而是还没有经过真实客户数据和大规模并发压测。内置评测能证明工程闭环，但不能包装成线上准确率。生产化需要引入真实审计样本、权限系统、数据库、对象存储、任务队列、模型网关和更严格的安全审计。

### 5.2 如果给你两周继续优化，你做什么？

优先级：

1. 增量评测和 badcase 自动归类。
2. 真实 MCP Server 接入示例。
3. 任务队列化，让评测和 Deep Research 后台运行。
4. 经验单元：把高价值失败样例沉淀到可检索知识。
5. 生产数据库迁移脚本。
6. 更细粒度 trace 可视化。

### 5.3 如果业务方说回答太慢怎么办？

先区分慢在哪里：路由、RAG、LLM、工具、评测、前端渲染。然后分层优化：

- 路由本地化，避免每次调大模型。
- RAG 做缓存、并行召回和 top-k 控制。
- Deep Research 走后台任务。
- 评测分 smoke/full 两层。
- 前端用折叠、局部刷新和命令中心减少操作成本。
- 对高频模板预生成和预热。

### 5.4 如果面试官质疑“没有训练模型”？

这个项目定位是 Agent 应用工程，不是基础模型训练平台。大厂 Agent 岗很多关注的是如何把模型接入真实业务流程：RAG、Tool Use、Memory、Evaluation、Guardrails、Observability。项目里保留了 SFT/RLHF 的离线入口，但我没有把未完成的训练包装成成果。当前重点是可运行、可评测、可治理的 Agent 系统。

## 6. 简历 Bullet 推荐

- 设计并实现面向 IT 审计交付的企业级 AI Agent 工作台，覆盖立项、取证、控制测试、风险评估、整改、人审和交付包下载闭环。
- 实现 Hybrid Intent Router，融合规则、n-gram 相似度、实体识别和紧急度信号，输出可解释路由、置信度和多 Agent 协作策略。
- 自研 Plan/Execute/Reflect Runtime，支持任务协议、依赖计划、角色步骤、工具调用、重试、反思、产物和运行指标持久化。
- 构建 Agentic RAG 管线，支持文档切块、持久化、多路查询改写、混合检索、来源追踪、置信度和无模型降级。
- 设计 Skills/MCP-style 工具治理层，提供输入 Schema、权限声明、TTL 缓存、熔断、调用日志和工具指标聚合。
- 建设 Agent/RAG/Deep Research 评测与发布门禁，支持历史基线对比、质量回归、延迟回归和 release blocker。
- 落地 Self-Evolution Harness，聚合评测退化、工具失败、记忆状态和能力缺口生成优化提案，并可物化为 Runtime 任务。
- 优化产品界面与信息架构，加入折叠记录、删除管理、全局命令中心、深链定位和低等待感交互。

## 7. 面试前 30 分钟速记

必须背熟：

- 项目不是 Chatbot，而是审计交付 Agent。
- 主链：Router → Memory → Runtime → Skills/RAG → Quality Gate → Evaluation。
- RAG：切块、查询扩展、混合检索、来源、置信度、降级。
- Tool：Schema、权限、缓存、熔断、日志。
- Eval：Agent/RAG/Research、基线、回归、发布门禁。
- 取舍：自研 Runtime 为可解释；生产可迁移 LangGraph/Temporal。
- 边界：没有线上压测就不吹线上准确率；SFT/RLHF 是离线方向。

## 8. 推荐演示路径

1. 打开首页，说明业务指标和交付就绪度。
2. 进入 `/audit` 创建或打开审计项目，展示控制矩阵、证据请求、整改任务、下载报告。
3. 进入 `/chat` 展示意图路由、多 Agent 协作和记忆。
4. 进入 `/knowledge` 展示 RAG 知识写入和查询。
5. 进入 `/skills` 展示 Runtime、Skill Registry、MCP、Harness、自进化任务。
6. 进入 `/training` 展示 Agent/RAG/Research 评测和发布门禁。
7. 用 `⌘K` 全局命令中心快速搜索评测或运行记录，证明不是静态页面。

## 9. 和面经高频点的映射表

| 面经高频点 | 项目材料中的回答位置 |
| --- | --- |
| Agent 架构 | 本文 3.2、`docs/01-项目架构与设计.md` |
| RAG 难点 | 本文 3.6、`docs/03-Agent核心能力详解.md` |
| Tool Use / MCP | 本文 3.8、`services/skill_registry.py` |
| 评测与回归 | 本文 3.10、`training/training_pipeline.py`、`services/evaluation_repository.py` |
| 自进化 Harness | 本文 3.11、`docs/10-Agent Harness与自进化市场雷达.md` |
| 项目真实性 | 本文 3.1、3.13、推荐演示路径 |
| 技术取舍 | 本文 3.5、3.12、3.13、3.15 |
| 性能优化 | 本文 3.14、5.3 |
| 业务价值 | 本文 1、2、8 |

## 10. 配套面经来源

完整 52 条来源索引见：

- `docs/interview_experience/2026-07-大厂Agent相关岗位面经50份索引.md`
- `docs/interview_experience/2026-07-大厂Agent面经逐条精读与题库.md`
- `docs/interview_experience/2026-07-大厂Agent面经公司专题与100问.md`
- 求职归档：`面经资料/01-大厂Agent相关岗位面经50份索引.md`
- 求职归档：`面经资料/02-大厂Agent面经逐条精读与题库.md`
- 求职归档：`面经资料/03-大厂Agent面经公司专题与100问.md`

## 11. 本轮细化后的重点背诵清单

### 11.1 Agent 开发岗必背

- Agent 和 Workflow 的区别。
- ReAct、Plan-and-Execute、Reflection 的区别。
- LangGraph 和自研 Runtime 的取舍。
- Multi-Agent 如何避免角色堆砌。
- 长周期任务如何 checkpoint 和恢复。
- Agent 失败如何定位：规划失败、检索失败、工具失败、生成失败、质量门失败。

### 11.2 RAG 必背

- 切块策略：固定长度、滑窗、段落、语义、递归切分。
- 检索策略：关键词、BM25/TF-IDF、向量、多路召回、混合检索。
- 错召处理：重排、过滤、权威性打分、质量门。
- 漏召处理：query rewrite、同义词、top-k、知识补全、GraphRAG。
- 评估指标：Recall@k、Precision@k、MRR、NDCG、faithfulness、source coverage。
- RAG vs 微调：知识更新与引用优先 RAG，稳定风格和能力迁移可微调。

### 11.3 Tool / MCP / Skill 必背

- Function Calling 是模型输出结构化工具调用。
- MCP 是连接 Host/Client/Server 的上下文协议。
- Skill 是工程治理单元，包含描述、Schema、权限、缓存、熔断和日志。
- 渐进式披露是按需暴露工具/技能描述，避免上下文噪声。
- 工具可靠性要看 schema validation、timeout、retry、circuit breaker、observability。

### 11.4 训练算法基础必背

- SFT：监督微调，学习高质量示例。
- LoRA：低秩适配，减少训练参数。
- RLHF/PPO：基于偏好和奖励模型做策略优化。
- DPO：直接偏好优化，不显式训练 reward model。
- GRPO：面向组内相对优势的强化学习优化思路。
- SFT 遗忘：混合通用数据、降低学习率、正则、分阶段训练、评测监控。

### 11.5 后端基础必背

- FastAPI：Pydantic、Depends、UploadFile、FileResponse、BackgroundTasks。
- 缓存：TTL、key 设计、失效策略、穿透/击穿/雪崩。
- 熔断：失败计数、open/half-open/closed、恢复窗口。
- 异步任务：评测、Deep Research、报告生成适合后台跑。
- 数据迁移：文件 Repository 到 DB/对象存储/日志系统的演进。

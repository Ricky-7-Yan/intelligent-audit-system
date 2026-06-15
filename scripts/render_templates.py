# -*- coding: utf-8 -*-
from pathlib import Path


def ent(text: str) -> str:
    return "".join(f"&#{ord(ch)};" if ord(ch) > 127 else ch for ch in text)


def nav(active: str, note: str) -> str:
    def cls(name: str) -> str:
        return " active" if name == active else ""

    return f"""
    <aside class="sidebar">
      <div class="brand"><div class="brand-icon">&#23457;</div><span>&#26234;&#33021;&#23457;&#35745;</span></div>
      <nav class="nav-list">
        <a class="nav-link{cls('home')}" href="/">&#8962; <span>&#24635;&#35272;</span></a>
        <a class="nav-link{cls('chat')}" href="/chat">&#9679; <span>&#23457;&#35745;&#23545;&#35805;</span></a>
        <a class="nav-link{cls('audit')}" href="/audit">&#8981; <span>&#23457;&#35745;&#20998;&#26512;</span></a>
        <a class="nav-link{cls('knowledge')}" href="/knowledge">&#9638; <span>&#30693;&#35782;&#24211;</span></a>
        <a class="nav-link{cls('training')}" href="/training">&#9633; <span>&#35780;&#20272;</span></a>
        <a class="nav-link{cls('skills')}" href="/skills">&#9671; <span>Skills</span></a>
      </nav>
      <div class="sidebar-note">{ent(note)}</div>
    </aside>
"""


def page(title: str, active: str, subtitle: str, body: str, note: str, scripts: str = "") -> str:
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{ent(title)} - {ent('智能审计 Agent 平台')}</title>
  <link rel="stylesheet" href="../static/app.css">
</head>
<body>
  <div class="app-shell">
{nav(active, note)}
    <main class="main">
      <div class="topbar">
        <div class="page-title">
          <h1>{ent(title)}</h1>
          <p>{ent(subtitle)}</p>
        </div>
        <div class="status-pill"><span class="status-dot"></span><span id="healthText">{ent('加载状态中')}</span></div>
      </div>
{body}
    </main>
  </div>
  <script src="../static/app.js"></script>
{scripts}
</body>
</html>
"""


def write(name: str, content: str) -> None:
    Path("templates", name).write_text(content, encoding="utf-8")


index_body = f"""
      <section class="kpi-row">
        <div class="card metric"><div class="metric-value" id="docCount">0</div><div class="metric-label">{ent('知识切片')}</div></div>
        <div class="card metric"><div class="metric-value" id="ragMode">-</div><div class="metric-label">{ent('检索模式')}</div></div>
        <div class="card metric"><div class="metric-value">7</div><div class="metric-label">{ent('内置控制项')}</div></div>
        <div class="card metric"><div class="metric-value">7</div><div class="metric-label">Skills</div></div>
        <div class="card metric"><div class="metric-value">2.4.0</div><div class="metric-label">{ent('当前版本')}</div></div>
      </section>
      <section class="grid grid-3 mt-16">
        <a class="panel" href="/audit"><h2 class="panel-title">{ent('审计分析工作台')}</h2><p class="muted">{ent('生成任务计划、证据包、控制矩阵、质量门、审计程序、抽样计划和整改任务。')}</p></a>
        <a class="panel" href="/skills"><h2 class="panel-title">Skills / MCP</h2><p class="muted">{ent('展示可注册、可描述、可审计的工具能力和 MCP 风格工具清单。')}</p></a>
        <a class="panel" href="/knowledge"><h2 class="panel-title">{ent('企业知识库')}</h2><p class="muted">{ent('沉淀制度、标准、底稿和流程文本，支撑可引用 RAG 回答。')}</p></a>
      </section>
      <section class="panel mt-16">
        <h2 class="panel-title">{ent('面向真实 JD 的能力覆盖')}</h2>
        <div class="grid grid-3">
          <div class="item"><strong>Agent Platform</strong><p class="muted">{ent('规划、工具调用、记忆、质量门、人工复核和可观测执行轨迹。')}</p></div>
          <div class="item"><strong>Skill / MCP</strong><p class="muted">{ent('Skill 注册、输入 Schema、权限声明、调用日志和 MCP 风格工具描述。')}</p></div>
          <div class="item"><strong>RAG Evaluation</strong><p class="muted">{ent('混合检索、来源引用、基准评测和企业审计知识增强。')}</p></div>
        </div>
      </section>
"""
index_scripts = f"""
  <script>
    apiFetch('/api/knowledge/stats').then((data) => {{
      const stats = data.stats || {{}};
      setText('#docCount', stats.total_documents || 0);
      setText('#ragMode', stats.semantic_retrieval ? '{ent('向量')}' : 'TF-IDF');
    }}).catch(() => {{}});
  </script>
"""
write("index.html", page("智能审计 Agent 平台", "home", "面向企业审计与大模型 Agent 岗位展示的端到端项目。", index_body, "企业级 Agent 能力：RAG、Skills、MCP、质量门和业务闭环。", index_scripts))


chat_body = f"""
      <div class="grid layout-2">
        <section class="panel">
          <div class="toolbar">
            <button class="btn" data-quick="{ent('请对ERP系统权限管理进行安全审计，参考ISO27001')}">ERP {ent('权限审计')}</button>
            <button class="btn" data-quick="{ent('检查财务报告流程是否符合SOX要求')}">SOX {ent('财务控制')}</button>
            <button class="btn" data-quick="{ent('分析数据备份流程的合规性和恢复风险')}">{ent('备份恢复风险')}</button>
          </div>
          <div class="chat-box mt-16" id="chatBox"><div class="message ai">{ent('你好，我是智能审计 Agent。请输入审计对象、标准或风险场景，我会给出结构化审计建议。')}</div></div>
          <div class="toolbar stretch mt-16"><input id="messageInput" placeholder="{ent('例如：请对ERP系统权限管理进行安全审计，参考ISO27001')}"><button class="btn primary" id="sendBtn">{ent('发送')}</button></div>
        </section>
        <aside class="grid">
          <section class="panel"><h2 class="panel-title">{ent('会话')}</h2><div class="list"><div class="item"><span class="muted">{ent('会话 ID')}</span><div id="sessionId"></div></div><div class="item"><span class="muted">{ent('消息数')}</span><div id="messageCount">0</div></div></div><button class="btn mt-12" id="clearBtn">{ent('清空对话')}</button></section>
          <section class="panel"><h2 class="panel-title">{ent('审计上下文')}</h2><div id="auditContext" class="list"><p class="muted">{ent('暂无上下文')}</p></div></section>
          <section class="panel"><h2 class="panel-title">{ent('风险与质量门')}</h2><div id="riskPanel" class="list"><p class="muted">{ent('暂无风险评估')}</p></div></section>
        </aside>
      </div>
"""
chat_scripts = """
  <script>
    let sessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    let messageCount = 0;
    setText("#sessionId", sessionId);
    function addMessage(content, type) { const box = qs("#chatBox"); box.appendChild(el("div", { class: `message ${type}`, text: content })); box.scrollTop = box.scrollHeight; }
    function renderContext(context) { const node = qs("#auditContext"); clearNode(node); Object.entries(context || {}).forEach(([key, value]) => node.appendChild(el("div", { class: "item" }, [el("strong", { text: key }), el("div", { class: "muted", text: Array.isArray(value) ? value.join("、") : String(value) })]))); if (!node.childElementCount) node.appendChild(el("p", { class: "muted", text: "暂无上下文" })); }
    function renderRisk(result) { const node = qs("#riskPanel"); clearNode(node); const risk = result.risk_assessment; const quality = result.quality_gate; if (!risk) return node.appendChild(el("p", { class: "muted", text: "暂无风险评估" })); node.appendChild(el("div", { class: "item" }, [riskBadge(risk.risk_level), el("div", { class: "mt-12", text: `风险评分：${risk.risk_score}` }), el("div", { class: "muted", text: `控制抵减：${risk.control_reduction || 0}` })])); if (quality) node.appendChild(el("div", { class: "item" }, [el("strong", { text: `质量门：${quality.status}` }), el("div", { class: "muted", text: `置信度：${quality.confidence}；缺失证据：${(quality.missing_evidence || []).join("、") || "无"}` })])); }
    async function sendMessage() { const input = qs("#messageInput"); const message = input.value.trim(); if (!message) return; addMessage(message, "user"); input.value = ""; addMessage("正在规划任务、检索证据、映射控制并评估风险...", "ai"); try { const data = await apiFetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message, session_id: sessionId }) }); qs("#chatBox").lastChild.remove(); addMessage(data.response, "ai"); renderContext(data.audit_context); renderRisk(data); messageCount += 2; setText("#messageCount", messageCount); } catch (error) { qs("#chatBox").lastChild.remove(); addMessage(`请求失败：${error.message}`, "ai"); } }
    qs("#sendBtn").addEventListener("click", sendMessage);
    qs("#messageInput").addEventListener("keydown", (event) => { if (event.key === "Enter") sendMessage(); });
    qs("#clearBtn").addEventListener("click", () => location.reload());
    qsa("[data-quick]").forEach((button) => button.addEventListener("click", () => { qs("#messageInput").value = button.dataset.quick; sendMessage(); }));
  </script>
"""
write("chat.html", page("审计对话", "chat", "围绕审计对象进行多轮分析，系统会同步给出风险、合规和整改信息。", chat_body, "对话结果同步输出上下文、风险、质量门和证据缺口。", chat_scripts))


knowledge_body = f"""
      <section class="grid layout-2">
        <div class="panel">
          <h2 class="panel-title">{ent('添加知识')}</h2>
          <div class="field"><label>{ent('知识内容')}</label><textarea id="knowledgeText" rows="8" placeholder="{ent('粘贴审计制度、控制描述、标准条款或审计底稿摘要')}"></textarea></div>
          <div class="grid grid-2 mt-12"><div class="field"><label>{ent('来源')}</label><input id="knowledgeSource" placeholder="{ent('例如：权限管理制度 V1.2')}"></div><div class="field"><label>{ent('类型')}</label><input id="knowledgeType" value="manual"></div></div>
          <button class="btn primary mt-16" id="addKnowledge">{ent('写入知识库')}</button><div id="addResult" class="muted mt-12"></div>
        </div>
        <div class="panel">
          <h2 class="panel-title">{ent('上传文本文件')}</h2><input type="file" id="fileInput" accept=".txt,.md,.csv,.json,.log"><button class="btn mt-16" id="uploadFile">{ent('上传并切块')}</button><div id="uploadResult" class="muted mt-12"></div>
          <div class="item mt-16"><strong>{ent('知识库统计')}</strong><p class="muted" id="statsText">{ent('加载中')}</p></div>
        </div>
      </section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('检索测试')}</h2><div class="toolbar stretch"><input id="searchQuery" placeholder="{ent('例如：SOX 财务系统内部控制重点')}"><button class="btn primary" id="searchBtn">{ent('检索')}</button></div><div id="searchResults" class="list mt-16"><p class="muted">{ent('输入问题后查看 RAG 答案和来源。')}</p></div></section>
"""
knowledge_scripts = """
  <script>
    async function refreshStats() { const data = await apiFetch("/api/knowledge/stats"); const stats = data.stats; setText("#statsText", `${stats.total_documents} 个切片 · ${stats.semantic_retrieval ? "语义向量" : "TF-IDF"} 检索 · 切块 ${stats.chunk_size}/${stats.chunk_overlap}`); }
    qs("#addKnowledge").addEventListener("click", async () => { const text = qs("#knowledgeText").value.trim(); if (!text) return setText("#addResult", "请输入知识内容"); try { const data = await apiFetch("/api/knowledge/add", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text, metadata: { source: qs("#knowledgeSource").value || "manual", type: qs("#knowledgeType").value || "manual" } }) }); setText("#addResult", `已新增 ${data.result.added_chunks} 个切片`); qs("#knowledgeText").value = ""; refreshStats(); } catch (error) { setText("#addResult", `写入失败：${error.message}`); } });
    qs("#uploadFile").addEventListener("click", async () => { const file = qs("#fileInput").files[0]; if (!file) return setText("#uploadResult", "请选择文件"); const form = new FormData(); form.append("file", file); try { const data = await apiFetch("/api/knowledge/upload", { method: "POST", body: form }); setText("#uploadResult", `已上传 ${data.file}，新增 ${data.result.added_chunks} 个切片`); refreshStats(); } catch (error) { setText("#uploadResult", `上传失败：${error.message}`); } });
    qs("#searchBtn").addEventListener("click", async () => { const question = qs("#searchQuery").value.trim(); if (!question) return; const node = qs("#searchResults"); clearNode(node); node.appendChild(el("p", { class: "muted", text: "检索中..." })); try { const data = await apiFetch(`/api/knowledge/query?question=${encodeURIComponent(question)}`); clearNode(node); node.appendChild(el("div", { class: "item prewrap", text: data.result.answer })); (data.result.sources || []).forEach((source) => node.appendChild(el("div", { class: "item" }, [el("strong", { text: source.source }), el("p", { class: "muted", text: `score ${source.score}` }), el("div", { text: source.content })]))); } catch (error) { clearNode(node); node.appendChild(el("p", { class: "muted", text: `检索失败：${error.message}` })); } });
    refreshStats().catch(() => setText("#statsText", "统计加载失败"));
  </script>
"""
write("knowledge.html", page("知识库", "knowledge", "维护审计制度、底稿、标准和流程文本，供 RAG 检索增强使用。", knowledge_body, "知识库支持制度、底稿、控制描述、标准条款和日志文本。", knowledge_scripts))


training_body = f"""
      <section class="panel"><h2 class="panel-title">{ent('评估任务')}</h2><p class="muted">{ent('运行轻量 Benchmark，评估 Agent 回答质量和 RAG 检索效果。')}</p><div class="toolbar"><button class="btn primary" id="runEval">{ent('运行 Agent 评估')}</button><button class="btn" id="runRagEval">RAG {ent('评测')}</button></div></section>
      <section class="kpi-row mt-16"><div class="card metric"><div class="metric-value" id="overallScore">-</div><div class="metric-label">{ent('总体得分')}</div></div><div class="card metric"><div class="metric-value" id="totalTests">-</div><div class="metric-label">{ent('测试用例')}</div></div><div class="card metric"><div class="metric-value">7</div><div class="metric-label">Skills</div></div><div class="card metric"><div class="metric-value">MCP</div><div class="metric-label">{ent('工具协议')}</div></div><div class="card metric"><div class="metric-value">2.4.0</div><div class="metric-label">{ent('架构版本')}</div></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('评估结果')}</h2><div id="evalResults" class="list"><p class="muted">{ent('尚未运行评估')}</p></div></section>
"""
training_scripts = """
  <script>
    function renderEvalResults(results) { const node = qs("#evalResults"); clearNode(node); Object.entries(results.overall_metrics?.category_scores || {}).forEach(([category, score]) => node.appendChild(el("div", { class: "item" }, [el("strong", { text: category }), el("div", { class: "muted", text: `得分：${score}` })]))); (results.results || []).forEach((item) => node.appendChild(el("div", { class: "item" }, [el("strong", { text: item.question }), el("p", { class: "muted", text: `类别：${item.category}` }), el("div", { class: "prewrap", text: item.actual_answer || JSON.stringify(item, null, 2) })]))); }
    qs("#runEval").addEventListener("click", async () => { const node = qs("#evalResults"); clearNode(node); node.appendChild(el("p", { class: "muted", text: "评估运行中..." })); try { const data = await apiFetch("/api/training/evaluate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model_path: "current-agent" }) }); const metrics = data.results.overall_metrics || {}; setText("#overallScore", metrics.overall_score ?? "-"); setText("#totalTests", metrics.total_tests ?? "-"); renderEvalResults(data.results); } catch (error) { clearNode(node); node.appendChild(el("p", { class: "muted", text: `评估失败：${error.message}` })); } });
    qs("#runRagEval").addEventListener("click", async () => { const node = qs("#evalResults"); clearNode(node); node.appendChild(el("p", { class: "muted", text: "RAG 评测运行中..." })); try { const data = await apiFetch("/api/evaluation/rag", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({}) }); setText("#overallScore", data.results.overall_score); setText("#totalTests", data.results.total_cases); clearNode(node); data.results.results.forEach((item) => node.appendChild(el("div", { class: "item" }, [el("strong", { text: item.question }), el("div", { class: "muted", text: `overall=${item.overall} · sources=${item.retrieved_docs_count}` })]))); } catch (error) { clearNode(node); node.appendChild(el("p", { class: "muted", text: `RAG 评测失败：${error.message}` })); } });
  </script>
"""
write("training.html", page("Agent 评估", "training", "评估当前审计 Agent 与 RAG 在治理、风险、合规和安全问题上的质量。", training_body, "线上运行轻量评估；大模型微调建议走离线任务。", training_scripts))


skills_body = f"""
      <section class="grid layout-2">
        <div class="panel"><h2 class="panel-title">Skill {ent('列表')}</h2><div id="skillList" class="list dense"><p class="muted">{ent('加载中')}</p></div></div>
        <div class="panel"><h2 class="panel-title">{ent('执行 Skill')}</h2><div class="field"><label>Skill {ent('名称')}</label><input id="skillName" value="audit.scope_planner"></div><div class="field mt-12"><label>{ent('输入 JSON')}</label><textarea id="skillInput" rows="8">{{"audit_item":"ERP系统权限管理","standard":"ISO27001","risk_topics":["权限","日志"]}}</textarea></div><button class="btn primary mt-16" id="runSkill">{ent('执行')}</button><pre id="skillOutput" class="item prewrap mt-16">{ent('等待执行')}</pre></div>
      </section>
      <section class="grid grid-2 mt-16"><div class="panel"><h2 class="panel-title">MCP Tools {ent('描述')}</h2><div id="mcpTools" class="list dense"><p class="muted">{ent('加载中')}</p></div></div><div class="panel"><h2 class="panel-title">Skill {ent('调用日志')}</h2><button class="btn" id="refreshRuns">{ent('刷新日志')}</button><div id="skillRuns" class="list dense mt-16"><p class="muted">{ent('暂无日志')}</p></div></div></section>
"""
skills_scripts = """
  <script>
    async function loadSkills() { const data = await apiFetch("/api/skills"); const node = qs("#skillList"); clearNode(node); data.skills.forEach((skill) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: `${skill.name} · ${skill.title}` }), el("p", { class: "muted", text: skill.description }), el("div", { text: `权限：${skill.permissions.join("、")}` })]))); }
    async function loadMcpTools() { const data = await apiFetch("/api/mcp/tools"); const node = qs("#mcpTools"); clearNode(node); data.tools.forEach((tool) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: tool.name }), el("p", { class: "muted", text: tool.description }), el("pre", { class: "prewrap", text: JSON.stringify(tool.inputSchema, null, 2) })]))); }
    async function loadRuns() { const data = await apiFetch("/api/skills/runs"); const node = qs("#skillRuns"); clearNode(node); if (!data.runs.length) return node.appendChild(el("p", { class: "muted", text: "暂无日志" })); data.runs.forEach((run) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: `${run.run_id} · ${run.skill} · ${run.status}` }), el("p", { class: "muted", text: run.finished_at }), el("pre", { class: "prewrap", text: JSON.stringify(run.output, null, 2) })]))); }
    qs("#runSkill").addEventListener("click", async () => { try { const input = JSON.parse(qs("#skillInput").value || "{}"); const name = qs("#skillName").value.trim(); const data = await apiFetch(`/api/skills/${encodeURIComponent(name)}/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input }) }); qs("#skillOutput").textContent = JSON.stringify(data.run, null, 2); loadRuns(); } catch (error) { qs("#skillOutput").textContent = `执行失败：${error.message}`; } });
    qs("#refreshRuns").addEventListener("click", loadRuns);
    loadSkills(); loadMcpTools(); loadRuns();
  </script>
"""
write("skills.html", page("Skills / MCP 工具层", "skills", "以可注册、可描述、可审计的方式管理 Agent 能力。", skills_body, "Skill 和 MCP 工具描述用于展示企业 Agent 平台化能力。", skills_scripts))


audit_body = f"""
      <section class="panel">
        <div class="grid grid-4">
          <div class="field"><label>{ent('审计对象')}</label><input id="auditItem" value="{ent('ERP系统权限管理')}"></div>
          <div class="field"><label>{ent('审计类型')}</label><select id="auditType"><option>{ent('安全审计')}</option><option>{ent('合规审计')}</option><option>{ent('风险评估')}</option><option>{ent('内部控制审计')}</option></select></div>
          <div class="field"><label>{ent('参考标准')}</label><select id="standardType"><option>ISO27001</option><option>COBIT</option><option>SOX</option><option>{ent('数据安全法')}</option></select></div>
          <div class="field"><label>{ent('关注风险')}</label><select id="riskLevel"><option>{ent('高')}</option><option>{ent('中')}</option><option>{ent('低')}</option></select></div>
        </div>
        <div class="toolbar mt-16"><button class="btn primary" id="runAudit">{ent('运行 Agent 审计')}</button><button class="btn" id="loadControls">{ent('查看控制库')}</button></div>
      </section>
      <section class="kpi-row mt-16">
        <div class="card metric"><div class="metric-value" id="riskLevelCard">-</div><div class="metric-label">{ent('剩余风险')}</div></div>
        <div class="card metric"><div class="metric-value" id="riskScoreCard">-</div><div class="metric-label">{ent('风险评分')}</div></div>
        <div class="card metric"><div class="metric-value" id="complianceCard">-</div><div class="metric-label">{ent('合规评分')}</div></div>
        <div class="card metric"><div class="metric-value" id="qualityCard">-</div><div class="metric-label">{ent('质量门置信度')}</div></div>
        <div class="card metric"><div class="metric-value" id="controlsCard">-</div><div class="metric-label">{ent('映射控制')}</div></div>
      </section>
      <section class="grid layout-2 mt-16"><div class="panel"><h2 class="panel-title">{ent('审计结论')}</h2><div id="auditResult" class="prewrap muted">{ent('等待分析结果')}</div></div><div class="panel"><h2 class="panel-title">{ent('质量门')}</h2><div id="qualityPanel" class="muted">{ent('等待结果')}</div></div></section>
      <section class="grid grid-2 mt-16"><div class="panel"><h2 class="panel-title">{ent('任务计划')}</h2><div id="taskPlan" class="timeline"><p class="muted">{ent('等待结果')}</p></div></div><div class="panel"><h2 class="panel-title">{ent('证据包')}</h2><div id="evidencePack" class="list dense"><p class="muted">{ent('等待结果')}</p></div></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('控制矩阵')}</h2><div id="controlMatrix" class="matrix muted">{ent('等待结果')}</div></section>
      <section class="grid grid-2 mt-16"><div class="panel"><h2 class="panel-title">{ent('审计程序')}</h2><div id="auditProgram" class="list dense"><p class="muted">{ent('等待结果')}</p></div></div><div class="panel"><h2 class="panel-title">{ent('抽样计划')}</h2><div id="samplingPlan" class="list dense"><p class="muted">{ent('等待结果')}</p></div></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('审计发现草稿')}</h2><div id="findings" class="grid grid-2"><p class="muted">{ent('等待结果')}</p></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('整改行动计划')}</h2><div id="recommendations" class="grid grid-3"><p class="muted">{ent('等待结果')}</p></div></section>
      <section class="panel mt-16"><h2 class="panel-title">{ent('整改任务跟踪')}</h2><div id="remediationTasks" class="list dense"><p class="muted">{ent('等待结果')}</p></div></section>
      <section class="grid layout-2 mt-16">
        <div class="panel"><h2 class="panel-title">{ent('审计档案')}</h2><div class="toolbar"><button class="btn" id="refreshRuns">{ent('刷新历史')}</button><a class="btn" id="downloadReport" href="#" target="_blank">{ent('下载当前报告')}</a></div><div id="runHistory" class="list dense mt-16"><p class="muted">{ent('暂无历史')}</p></div></div>
        <div class="panel"><h2 class="panel-title">{ent('人工复核')}</h2><div class="grid grid-2"><div class="field"><label>{ent('复核人')}</label><input id="reviewer" value="{ent('审计经理')}"></div><div class="field"><label>{ent('结论')}</label><select id="reviewDecision"><option value="approve">{ent('通过')}</option><option value="need_evidence">{ent('补充证据')}</option><option value="reject">{ent('退回整改')}</option></select></div></div><div class="field mt-12"><label>{ent('复核意见')}</label><textarea id="reviewComment" rows="4" placeholder="{ent('记录人工判断、证据缺口或整改要求')}"></textarea></div><button class="btn primary mt-16" id="submitReview">{ent('提交复核')}</button><div id="reviewResult" class="muted mt-12"></div></div>
      </section>
"""
audit_scripts = """
  <script>
    let currentRunId = null;
    function statusBadge(status) { return el("span", { class: `badge ${status}`, text: status }); }
    function renderTaskPlan(tasks) { const node = qs("#taskPlan"); clearNode(node); tasks.forEach((task, index) => node.appendChild(el("div", { class: "timeline-step" }, [el("div", { class: "step-index", text: String(index + 1) }), el("div", { class: "item compact" }, [el("strong", { text: task.name }), el("div", { class: "muted", text: task.objective }), el("div", { class: "mt-12", text: `负责人：${task.owner}` })])]))); }
    function renderEvidence(items) { const node = qs("#evidencePack"); clearNode(node); items.forEach((item) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: `${item.id} · ${item.source}` }), el("p", { class: "muted", text: item.summary }), el("div", { text: `用途：${item.usage}` })]))); }
    function renderQuality(quality) { const node = qs("#qualityPanel"); clearNode(node); const percent = Math.round((quality.confidence || 0) * 100); const ring = el("div", { class: "quality-ring" }, [el("span", { text: `${percent}%` })]); ring.style.setProperty("--score", percent); node.appendChild(el("div", { class: "quality" }, [ring, el("div", {}, [statusBadge(quality.status || "review"), el("p", { class: "muted", text: quality.review_note || "" }), el("div", { text: `证据扎实度：${quality.groundedness || 0} · 控制覆盖：${quality.control_coverage || 0}` }), el("div", { class: "mt-12 muted", text: `缺失证据：${(quality.missing_evidence || []).join("、") || "无"}` })])]))); }
    function renderMatrix(rows) { const node = qs("#controlMatrix"); clearNode(node); const table = el("table"); table.appendChild(el("thead", {}, [el("tr", {}, ["控制", "领域", "测试程序", "证据要求", "成熟度", "状态"].map((text) => el("th", { text })))])); const body = el("tbody"); rows.forEach((row) => body.appendChild(el("tr", {}, [el("td", { text: row.control_id || row.id }), el("td", { text: row.domain }), el("td", { text: row.test_procedure }), el("td", { text: (row.evidence_required || []).join("、") }), el("td", { text: String(row.maturity_level ?? "-") }), el("td", { text: row.status || row.objective || "" })]))); table.appendChild(body); node.appendChild(table); }
    function renderAuditProgram(items) { const node = qs("#auditProgram"); clearNode(node); if (!items.length) return node.appendChild(el("p", { class: "muted", text: "暂无审计程序" })); items.forEach((item) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: `${item.step_id} · ${item.control_id}` }), el("div", { class: "muted", text: item.procedure }), el("div", { class: "mt-12", text: `认定：${item.assertion}` }), el("div", { class: "muted", text: `底稿：${item.workpaper_ref}` })]))); }
    function renderSamplingPlan(plan) { const node = qs("#samplingPlan"); clearNode(node); if (!plan || !plan.population) return node.appendChild(el("p", { class: "muted", text: "暂无抽样计划" })); [["总体", plan.population], ["期间", plan.period], ["方法", plan.method], ["样本量", plan.sample_size], ["分层", (plan.strata || []).join("、")], ["例外处理", plan.exception_handling]].forEach(([label, value]) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: label }), el("div", { class: "muted", text: String(value || "") })]))); }
    function renderFindings(items) { const node = qs("#findings"); clearNode(node); if (!items.length) return node.appendChild(el("p", { class: "muted", text: "当前未形成重大审计发现草稿" })); items.forEach((finding) => node.appendChild(el("div", { class: "card" }, [el("strong", { text: `${finding.finding_id} · ${finding.title}` }), el("p", { class: "muted", text: `严重程度：${finding.severity}` }), el("div", { text: finding.condition }), el("p", { class: "muted", text: `影响：${finding.effect}` }), el("div", { text: `建议：${finding.recommendation}` })]))); }
    function renderRecommendations(items) { const node = qs("#recommendations"); clearNode(node); items.forEach((rec) => node.appendChild(el("div", { class: "card" }, [el("strong", { text: `${rec.type} · ${rec.priority}` }), el("p", { class: "muted", text: rec.description }), el("div", { text: rec.action_items.join("；") }), el("div", { class: "mt-12 muted", text: `责任角色：${rec.owner_role} · ${rec.due_days}天 · ${rec.success_metric}` })]))); }
    function renderTasks(tasks) { const node = qs("#remediationTasks"); clearNode(node); if (!tasks || !tasks.length) return node.appendChild(el("p", { class: "muted", text: "暂无整改任务" })); tasks.forEach((task) => { const status = el("select"); ["未开始", "进行中", "待验证", "已完成", "已关闭"].forEach((value) => { const option = el("option", { value, text: value }); if (value === task.status) option.selected = true; status.appendChild(option); }); const owner = el("input", { value: task.owner || "", placeholder: "责任人" }); const note = el("input", { placeholder: "更新备注" }); const save = el("button", { class: "btn", text: "更新" }); save.addEventListener("click", () => updateTask(task.task_id, status.value, owner.value, note.value)); node.appendChild(el("div", { class: "item" }, [el("strong", { text: `${task.task_id} · ${task.title} · ${task.priority}` }), el("p", { class: "muted", text: task.description }), el("div", { text: `验收指标：${task.success_metric}` }), el("div", { class: "grid grid-4 mt-12" }, [status, owner, note, save])])); }); }
    function renderResult(result) { setText("#auditResult", result.response); setText("#riskLevelCard", result.risk_assessment.risk_level); setText("#riskScoreCard", result.risk_assessment.risk_score); setText("#complianceCard", result.compliance_check.compliance_score); setText("#qualityCard", result.quality_gate.confidence); setText("#controlsCard", result.control_matrix.length); renderTaskPlan(result.task_plan || []); renderEvidence(result.evidence_pack || []); renderQuality(result.quality_gate || {}); renderMatrix(result.control_matrix || []); renderAuditProgram(result.audit_program || []); renderSamplingPlan(result.sampling_plan || {}); renderFindings(result.findings || []); renderRecommendations(result.recommendations || []); }
    function renderRunRecord(record) { renderResult(record.result); renderTasks(record.remediation_tasks || []); }
    async function loadRuns() { const node = qs("#runHistory"); clearNode(node); try { const data = await apiFetch("/api/audit/runs?limit=10"); if (!data.runs.length) return node.appendChild(el("p", { class: "muted", text: "暂无历史" })); data.runs.forEach((run) => { const item = el("div", { class: "item compact" }, [el("strong", { text: `${run.run_id} · ${run.audit_item || ""}` }), el("div", { class: "muted", text: `${run.status} · 风险 ${run.risk_level || "-"} · 合规 ${run.compliance_score ?? "-"} · 质量 ${run.quality_confidence ?? "-"}` }), el("div", { class: "mt-12", text: run.created_at || "" })]); item.addEventListener("click", () => loadRunDetail(run.run_id)); node.appendChild(item); }); } catch (error) { node.appendChild(el("p", { class: "muted", text: `加载失败：${error.message}` })); } }
    async function loadRunDetail(runId) { const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(runId)}`); currentRunId = runId; qs("#downloadReport").href = serviceUrl(`/api/audit/runs/${encodeURIComponent(runId)}/report.md`); renderRunRecord(data.run); }
    qs("#runAudit").addEventListener("click", async () => { setText("#auditResult", "正在运行 Agent 工作流..."); try { const data = await apiFetch("/api/audit", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ audit_item: qs("#auditItem").value, audit_type: qs("#auditType").value, standard_type: qs("#standardType").value, risk_level: qs("#riskLevel").value }) }); currentRunId = data.run_id; qs("#downloadReport").href = serviceUrl(`/api/audit/runs/${encodeURIComponent(currentRunId)}/report.md`); renderResult(data.result); loadRunDetail(currentRunId); loadRuns(); } catch (error) { setText("#auditResult", `分析失败：${error.message}`); } });
    qs("#loadControls").addEventListener("click", async () => { const data = await apiFetch("/api/audit/controls"); renderMatrix(data.controls); });
    qs("#refreshRuns").addEventListener("click", loadRuns);
    qs("#submitReview").addEventListener("click", async () => { if (!currentRunId) return setText("#reviewResult", "请先运行或选择一条审计档案"); try { await apiFetch(`/api/audit/runs/${encodeURIComponent(currentRunId)}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ reviewer: qs("#reviewer").value, decision: qs("#reviewDecision").value, comment: qs("#reviewComment").value }) }); setText("#reviewResult", "复核已保存"); loadRuns(); } catch (error) { setText("#reviewResult", `复核失败：${error.message}`); } });
    async function updateTask(taskId, status, owner, note) { if (!currentRunId) return; try { const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(currentRunId)}/tasks/${encodeURIComponent(taskId)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ status, owner, note }) }); renderRunRecord(data.run); loadRuns(); } catch (error) { setText("#reviewResult", `任务更新失败：${error.message}`); } }
    loadRuns();
  </script>
"""
write("audit.html", page("审计分析", "audit", "将审计对象转化为控制矩阵、证据包、风险结论、审计程序、抽样计划和整改闭环。", audit_body, "Agent 工作流：任务规划、证据检索、控制映射、风险评分、质量门、整改计划。", audit_scripts))

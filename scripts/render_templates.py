# -*- coding: utf-8 -*-
from pathlib import Path


T = {
    "app": "\u667a\u80fd\u5ba1\u8ba1 Agent \u5e73\u53f0",
    "home": "\u603b\u89c8",
    "chat": "\u5ba1\u8ba1\u5bf9\u8bdd",
    "audit": "\u5ba1\u8ba1\u5206\u6790",
    "knowledge": "\u77e5\u8bc6\u5e93",
    "training": "\u8bc4\u4f30",
    "loading": "\u52a0\u8f7d\u72b6\u6001\u4e2d",
}


def ent(text: str) -> str:
    return "".join(f"&#{ord(ch)};" if ord(ch) > 127 else ch for ch in text)


def nav(active: str, note: str) -> str:
    def cls(name: str) -> str:
        return " active" if name == active else ""

    return f"""
    <aside class="sidebar">
      <div class="brand"><div class="brand-icon">&#23457;</div><span>{ent(T['app'])}</span></div>
      <nav class="nav-list">
        <a class="nav-link{cls('home')}" href="/">&#8962; <span>{ent(T['home'])}</span></a>
        <a class="nav-link{cls('chat')}" href="/chat">&#9679; <span>{ent(T['chat'])}</span></a>
        <a class="nav-link{cls('audit')}" href="/audit">&#8981; <span>{ent(T['audit'])}</span></a>
        <a class="nav-link{cls('knowledge')}" href="/knowledge">&#9638; <span>{ent(T['knowledge'])}</span></a>
        <a class="nav-link{cls('training')}" href="/training">&#9633; <span>{ent(T['training'])}</span></a>
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
  <title>{ent(title)} - {ent(T['app'])}</title>
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
        <div class="status-pill"><span class="status-dot"></span><span id="healthText">{ent(T['loading'])}</span></div>
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
      <section class="hero-console">
        <div class="panel console-hero">
          <div>
            <div class="hero-eyebrow">Audit Operations Console</div>
            <h2 class="hero-title">{ent('把审计从一次性问答升级为可运营的企业 Agent 工作台')}</h2>
            <p class="hero-copy">{ent('覆盖范围规划、证据检索、控制映射、风险评分、质量门、人工复核和整改闭环。所有关键动作都有状态、来源、负责人和可追溯记录。')}</p>
            <div class="hero-actions">
              <a class="btn primary" href="/audit">{ent('启动审计')}</a>
              <a class="btn" href="/knowledge">{ent('管理知识库')}</a>
              <a class="btn" href="/skills">{ent('查看 Agent 能力')}</a>
            </div>
          </div>
          <div class="kpi-row mt-16">
            <div class="card metric"><div class="metric-value" id="runCount">0</div><div class="metric-label">{ent('审计档案')}</div></div>
            <div class="card metric"><div class="metric-value" id="taskCount">0</div><div class="metric-label">{ent('待办整改')}</div></div>
            <div class="card metric"><div class="metric-value" id="qualityAvg">0</div><div class="metric-label">{ent('平均质量')}</div></div>
            <div class="card metric"><div class="metric-value" id="docCount">0</div><div class="metric-label">{ent('知识切片')}</div></div>
            <div class="card metric"><div class="metric-value" id="skillCount">7</div><div class="metric-label">Skills</div></div>
          </div>
        </div>
        <aside class="panel">
          <h2 class="panel-title">{ent('运营信号')}</h2>
          <div id="signals" class="signal-list"></div>
        </aside>
      </section>

      <div class="section-title"><div><h2>{ent('Agent 执行流水线')}</h2><p>{ent('把成熟审计方法固化为可观察、可复核、可扩展的工作流。')}</p></div></div>
      <section id="pipeline" class="pipeline-grid"></section>

      <div class="section-title"><div><h2>{ent('企业连接与能力层')}</h2><p>{ent('知识库、审计档案、Skill Registry、模型网关和可选数据源的运行状态。')}</p></div></div>
      <section id="connectors" class="connector-grid"></section>

      <div class="section-title"><div><h2>{ent('最近审计')}</h2><p>{ent('客户工作台需要能快速回到最新档案和风险状态。')}</p></div><a class="btn" href="/audit">{ent('进入工作台')}</a></div>
      <section class="panel"><div id="recentRuns" class="list dense"></div></section>

      <div class="section-title"><div><h2>{ent('客户价值')}</h2><p>{ent('产品需要回答为什么企业愿意把它用于真实审计流程。')}</p></div></div>
      <section id="customerValue" class="grid grid-4"></section>
"""

index_scripts = """
  <script>
    function pct(value) { return `${Math.round((Number(value) || 0) * 100)}%`; }
    function renderOverview(data) {
      const overview = data.overview || {};
      const summary = overview.summary || {};
      setText("#runCount", summary.audit_runs || 0);
      setText("#taskCount", summary.open_tasks || 0);
      setText("#qualityAvg", summary.avg_quality ? pct(summary.avg_quality) : "0%");
      setText("#docCount", summary.knowledge_chunks || 0);
      setText("#skillCount", summary.skills || 0);

      const signals = qs("#signals"); clearNode(signals);
      [
        ["知识库", `${summary.knowledge_chunks || 0} 个切片可检索`, "ok"],
        ["整改", `${summary.open_tasks || 0} 个开放任务`, summary.overdue_tasks ? "danger" : "warning"],
        ["质量门", `平均置信度 ${summary.avg_quality ? pct(summary.avg_quality) : "0%"}`, "ok"],
        ["合规", `平均评分 ${summary.avg_compliance || 0}`, "ok"],
      ].forEach(([name, detail, tone]) => signals.appendChild(el("div", { class: `signal ${tone}` }, [el("span", { class: "signal-dot" }), el("div", {}, [el("strong", { text: name }), el("div", { class: "muted", text: detail })]), el("span", { class: "status-chip", text: tone === "danger" ? "需关注" : "正常" })])));

      const pipeline = qs("#pipeline"); clearNode(pipeline);
      (overview.pipeline || []).forEach((step) => pipeline.appendChild(el("div", { class: "card pipeline-step" }, [el("div", { class: "step-code", text: step.stage }), el("h3", { class: "panel-title", text: step.title }), el("p", { class: "muted", text: step.detail })])));

      const connectors = qs("#connectors"); clearNode(connectors);
      (overview.connectors || []).forEach((item) => connectors.appendChild(el("div", { class: "card connector" }, [el("div", { class: "connector-head" }, [el("strong", { text: item.name }), el("span", { class: `status-chip ${item.status}`, text: item.status })]), el("div", { class: "muted", text: item.detail })])));

      const runs = qs("#recentRuns"); clearNode(runs);
      if (!(overview.recent_runs || []).length) runs.appendChild(el("p", { class: "muted", text: "暂无审计档案，先运行一次审计分析。" }));
      (overview.recent_runs || []).forEach((run) => runs.appendChild(el("div", { class: "item run-row clickable", onclick: () => { location.href = "/audit"; } }, [el("strong", { text: run.audit_item || run.run_id }), riskBadge(run.risk_level), el("div", { class: "muted", text: `质量 ${run.quality_confidence ?? "-"}` }), el("div", { class: "muted", text: run.status || "-" })])));

      const values = qs("#customerValue"); clearNode(values);
      (overview.customer_value || []).forEach((item) => values.appendChild(el("div", { class: "card value-card" }, [el("h3", { class: "panel-title", text: item.title }), el("p", { class: "muted", text: item.detail })])));
    }
    apiFetch("/api/product/overview").then(renderOverview).catch(() => {});
  </script>
"""

write("index.html", page(T["app"], "home", "\u9762\u5411\u5ba2\u6237\u4f7f\u7528\u7684\u4f01\u4e1a\u5ba1\u8ba1 Agent \u8fd0\u8425\u5e73\u53f0", index_body, "\u4ea7\u54c1\u5316\u80fd\u529b\uff1a\u5ba1\u8ba1\u8fd0\u8425\u3001RAG\u3001Skills\u3001MCP\u3001\u8d28\u91cf\u95e8\u548c\u6574\u6539\u95ed\u73af\u3002", index_scripts))


skills_body = f"""
      <section class="hero-console">
        <div class="panel console-hero">
          <div>
            <div class="hero-eyebrow">Agent Capability Center</div>
            <h2 class="hero-title">{ent('把 Agent 能力做成可治理的企业工具层')}</h2>
            <p class="hero-copy">{ent('每个 Skill 都包含输入 Schema、权限声明、版本和调用日志；MCP 风格描述让工具能力可以被 Agent 客户端发现和编排。')}</p>
          </div>
          <div class="kpi-row mt-16">
            <div class="card metric"><div class="metric-value" id="skillTotal">0</div><div class="metric-label">Skills</div></div>
            <div class="card metric"><div class="metric-value" id="toolTotal">0</div><div class="metric-label">MCP Tools</div></div>
            <div class="card metric"><div class="metric-value" id="runTotal">0</div><div class="metric-label">{ent('调用记录')}</div></div>
            <div class="card metric"><div class="metric-value">Schema</div><div class="metric-label">{ent('输入约束')}</div></div>
            <div class="card metric"><div class="metric-value">Audit</div><div class="metric-label">{ent('可追溯')}</div></div>
          </div>
        </div>
        <div class="panel">
          <h2 class="panel-title">{ent('执行 Skill')}</h2>
          <div class="field"><label>Skill {ent('名称')}</label><input id="skillName" value="audit.control_mapper"></div>
          <div class="field mt-12"><label>{ent('输入 JSON')}</label><textarea id="skillInput" rows="9">{{"audit_item":"ERP系统权限管理","standard":"ISO27001","risk_topics":["权限","日志","变更"]}}</textarea></div>
          <button class="btn primary mt-16" id="runSkill">{ent('执行')}</button>
        </div>
      </section>

      <div class="section-title"><div><h2>Skill Registry</h2><p>{ent('企业 Agent 平台需要可发现、可审计、可灰度扩展的工具能力。')}</p></div></div>
      <section id="skillList" class="connector-grid"></section>

      <div class="section-title"><div><h2>MCP Tools</h2><p>{ent('面向工具协议的描述层，包含 inputSchema 与权限注解。')}</p></div></div>
      <section id="mcpTools" class="grid grid-2"></section>

      <div class="section-title"><div><h2>{ent('执行结果与日志')}</h2><p>{ent('客户产品里，任何自动化动作都应可追踪、可解释、可复盘。')}</p></div><button class="btn" id="refreshRuns">{ent('刷新日志')}</button></div>
      <section class="grid layout-2">
        <pre id="skillOutput" class="item prewrap">{ent('等待执行')}</pre>
        <div id="skillRuns" class="list dense"></div>
      </section>
"""

skills_scripts = """
  <script>
    async function loadSkills() { const data = await apiFetch("/api/skills"); setText("#skillTotal", data.skills.length); const node = qs("#skillList"); clearNode(node); data.skills.forEach((skill) => node.appendChild(el("div", { class: "card connector" }, [el("div", { class: "connector-head" }, [el("strong", { text: skill.title }), el("span", { class: "status-chip", text: skill.version })]), el("div", { class: "muted", text: skill.name }), el("p", { class: "muted", text: skill.description }), el("div", { text: `权限：${skill.permissions.join("、")}` })]))); }
    async function loadMcpTools() { const data = await apiFetch("/api/mcp/tools"); setText("#toolTotal", data.tools.length); const node = qs("#mcpTools"); clearNode(node); data.tools.forEach((tool) => node.appendChild(el("div", { class: "card" }, [el("h3", { class: "panel-title", text: tool.name }), el("p", { class: "muted", text: tool.description }), el("pre", { class: "prewrap", text: JSON.stringify(tool.inputSchema, null, 2) })]))); }
    async function loadRuns() { const data = await apiFetch("/api/skills/runs"); setText("#runTotal", data.runs.length); const node = qs("#skillRuns"); clearNode(node); if (!data.runs.length) return node.appendChild(el("p", { class: "muted", text: "暂无日志" })); data.runs.forEach((run) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: `${run.run_id} · ${run.skill} · ${run.status}` }), el("p", { class: "muted", text: run.finished_at }), el("pre", { class: "prewrap", text: JSON.stringify(run.output, null, 2) })]))); }
    qs("#runSkill").addEventListener("click", async () => { try { const input = JSON.parse(qs("#skillInput").value || "{}"); const name = qs("#skillName").value.trim(); const data = await apiFetch(`/api/skills/${encodeURIComponent(name)}/run`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ input }) }); qs("#skillOutput").textContent = JSON.stringify(data.run, null, 2); loadRuns(); } catch (error) { qs("#skillOutput").textContent = `执行失败：${error.message}`; } });
    qs("#refreshRuns").addEventListener("click", loadRuns);
    loadSkills(); loadMcpTools(); loadRuns();
  </script>
"""

write("skills.html", page("Skills / MCP 工具层", "skills", "企业 Agent 的能力中心、工具协议层和调用审计台。", skills_body, "成熟产品需要把 Agent 能力从代码函数升级为可治理的工具资产。", skills_scripts))

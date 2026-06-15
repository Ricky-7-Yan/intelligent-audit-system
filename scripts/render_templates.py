# -*- coding: utf-8 -*-
"""Render product pages that are easier to keep encoding-safe."""

from pathlib import Path


def ent(text: str) -> str:
    return "".join(f"&#{ord(ch)};" if ord(ch) > 127 else ch for ch in text)


def page(title: str, active: str, heading: str, body: str, subtitle: str = "") -> str:
    nav = [
        ("/", "home", "总览", "⌂"),
        ("/chat", "chat", "审计对话", "●"),
        ("/audit", "audit", "审计工作台", "⌕"),
        ("/knowledge", "knowledge", "知识库", "▦"),
        ("/training", "training", "评测", "□"),
        ("/skills", "skills", "Skills", "◇"),
    ]
    links = "\n".join(
        f'<a class="nav-link {"active" if key == active else ""}" href="{href}">{icon} <span>{ent(label)}</span></a>'
        for href, key, label, icon in nav
    )
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{ent(title)}</title>
  <link rel="stylesheet" href="../static/app.css">
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-icon">AP</div><span>{ent('审脉 AuditPilot')}</span></div>
      <nav class="nav-list">{links}</nav>
      <div class="sidebar-note">{ent('以审计交付为主线：知识检索、控制测试、底稿、复核和整改闭环。')}</div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div class="page-title"><h1>{ent(heading)}</h1><p>{ent(subtitle)}</p></div>
        <div class="status-pill"><span class="status-dot"></span><span id="healthText">{ent('加载状态中')}</span></div>
      </div>
      {body}
    </main>
  </div>
  <script src="../static/app.js"></script>
</body>
</html>
"""


index_body = f"""
<section class="hero-console">
  <div class="panel console-hero">
    <div>
      <div class="hero-eyebrow">{ent('Enterprise Audit Agent')}</div>
      <h2 class="hero-title">{ent('面向审计行业交付的企业级 Agent 工作台')}</h2>
      <p class="hero-copy">{ent('把审计范围、证据请求、控制测试、抽样底稿、审计发现、人工复核和整改跟踪放在同一条可追溯链路上，帮助审计团队从问答演示走向真实项目交付。')}</p>
      <div class="hero-actions">
        <a class="btn primary" href="/audit">{ent('启动审计项目')}</a>
        <a class="btn" href="/knowledge">{ent('维护知识库')}</a>
        <a class="btn" href="/skills">{ent('查看 Skills / MCP')}</a>
      </div>
    </div>
    <div class="kpi-row">
      <div class="card metric"><div class="metric-value" id="runCount">0</div><div class="metric-label">{ent('审计档案')}</div></div>
      <div class="card metric"><div class="metric-value" id="openTasks">0</div><div class="metric-label">{ent('开放整改')}</div></div>
      <div class="card metric"><div class="metric-value" id="avgCompliance">0</div><div class="metric-label">{ent('平均合规')}</div></div>
      <div class="card metric"><div class="metric-value" id="knowledgeChunks">0</div><div class="metric-label">{ent('知识切片')}</div></div>
      <div class="card metric"><div class="metric-value" id="skillCount">0</div><div class="metric-label">Skills</div></div>
    </div>
  </div>
  <aside class="panel">
    <h2 class="panel-title">{ent('交付信号')}</h2>
    <div class="signal-list" id="signals"><p class="muted">{ent('加载中')}</p></div>
  </aside>
</section>
<section class="section-title"><div><h2>{ent('风险台账')}</h2><p>{ent('成熟审计产品需要直接管理风险、责任人、状态和下一步动作。')}</p></div></section>
<section class="grid grid-3" id="riskRegister"></section>
<section class="section-title"><div><h2>{ent('证据请求队列')}</h2><p>{ent('把缺失证据变成可跟踪的业务请求，减少线下追问。')}</p></div></section>
<section class="grid grid-3" id="evidenceRequests"></section>
<section class="section-title"><div><h2>{ent('控制健康')}</h2><p>{ent('按控制领域聚合成熟度、例外和风险热区。')}</p></div></section>
<section class="connector-grid" id="controlHealth"></section>
<script>
async function loadOverview() {{
  const overview = (await apiFetch("/api/product/overview")).overview;
  const summary = overview.summary || {{}};
  setText("#runCount", summary.audit_runs || 0);
  setText("#openTasks", summary.open_tasks || 0);
  setText("#avgCompliance", summary.avg_compliance || 0);
  setText("#knowledgeChunks", summary.knowledge_chunks || 0);
  setText("#skillCount", summary.skills || 0);
  const signals = qs("#signals"); clearNode(signals);
  [
    ["知识库", `${{summary.knowledge_chunks || 0}} 个切片可检索`, "ok"],
    ["整改", `${{summary.open_tasks || 0}} 个开放任务`, summary.overdue_tasks ? "danger" : "warning"],
    ["质量", `平均置信度 ${{summary.avg_quality || 0}}`, "ok"]
  ].forEach(([name, detail, tone]) => signals.appendChild(el("div", {{ class: `signal ${{tone}}` }}, [el("span", {{ class: "signal-dot" }}), el("div", {{}}, [el("strong", {{ text: name }}), el("div", {{ class: "muted", text: detail }})]), el("span", {{ class: "status-chip", text: tone === "danger" ? "需关注" : "正常" }})])));
  const risk = qs("#riskRegister"); clearNode(risk);
  (overview.risk_register || []).slice(0, 6).forEach((item) => risk.appendChild(el("div", {{ class: "card value-card" }}, [el("strong", {{ text: `${{item.risk_id}} · ${{item.audit_item}}` }}), el("p", {{ class: "muted", text: `风险：${{item.risk_level}} · 状态：${{item.status}}` }}), el("div", {{ text: item.next_action }})])));
  if (!(overview.risk_register || []).length) risk.appendChild(el("div", {{ class: "card value-card" }}, [el("h3", {{ class: "panel-title", text: "暂无风险台账" }}), el("p", {{ class: "muted", text: "运行审计项目后自动生成。" }})]));
  const evidence = qs("#evidenceRequests"); clearNode(evidence);
  (overview.evidence_requests || []).slice(0, 6).forEach((item) => evidence.appendChild(el("div", {{ class: "card value-card" }}, [el("strong", {{ text: item.evidence }}), el("p", {{ class: "muted", text: `${{item.audit_item}} · ${{item.owner}} · ${{item.priority}}` }}), el("div", {{ text: item.status }})])));
  if (!(overview.evidence_requests || []).length) evidence.appendChild(el("div", {{ class: "card value-card" }}, [el("h3", {{ class: "panel-title", text: "暂无证据请求" }}), el("p", {{ class: "muted", text: "质量门通过时不会产生补证队列。" }})]));
  const health = qs("#controlHealth"); clearNode(health);
  (overview.control_health || []).slice(0, 6).forEach((item) => health.appendChild(el("div", {{ class: "card connector" }}, [el("strong", {{ text: item.domain }}), el("div", {{ class: "muted", text: `控制 ${{item.controls}} · 成熟度 ${{item.avg_maturity}} · 例外 ${{item.exceptions}}` }}), el("div", {{ class: "progress-track" }}, [el("div", {{ class: "progress-bar", style: `width:${{Math.min(item.health_score || 0, 100)}}%` }})])])));
  if (!(overview.control_health || []).length) health.appendChild(el("div", {{ class: "card connector" }}, [el("strong", {{ text: "暂无控制健康数据" }}), el("div", {{ class: "muted", text: "运行审计后自动聚合控制领域。" }})]));
}}
document.addEventListener("DOMContentLoaded", loadOverview);
</script>
"""


skills_body = f"""
<section class="hero-console">
  <div class="panel console-hero">
    <div>
      <div class="hero-eyebrow">Skill / MCP</div>
      <h2 class="hero-title">{ent('企业 Agent 的能力中心与工具治理层')}</h2>
      <p class="hero-copy">{ent('每个 Skill 都包含输入 Schema、权限声明、版本和调用日志；MCP 风格描述让工具能力可以被 Agent 客户端发现、编排和审计。')}</p>
    </div>
  </div>
  <aside class="panel">
    <h2 class="panel-title">{ent('工具运行原则')}</h2>
    <div class="signal-list">
      <div class="signal"><span class="signal-dot"></span><div><strong>{ent('可发现')}</strong><div class="muted">{ent('标准化工具描述与输入结构')}</div></div><span class="status-chip">{ent('治理')}</span></div>
      <div class="signal"><span class="signal-dot"></span><div><strong>{ent('可审计')}</strong><div class="muted">{ent('每次调用写入运行日志')}</div></div><span class="status-chip">{ent('留痕')}</span></div>
      <div class="signal"><span class="signal-dot"></span><div><strong>{ent('可扩展')}</strong><div class="muted">{ent('面向审计流程沉淀专业能力')}</div></div><span class="status-chip">{ent('扩展')}</span></div>
    </div>
  </aside>
</section>
<div class="section-title"><div><h2>Skill Registry</h2><p>{ent('可治理、可发现、可审计的审计工具能力。')}</p></div></div>
<section class="grid grid-3" id="skillGrid"></section>
<div class="section-title"><div><h2>MCP Tools</h2><p>{ent('面向工具协议的描述层，包含 inputSchema 与权限注解。')}</p></div></div>
<section class="matrix" id="mcpTools"></section>
<div class="section-title"><div><h2>{ent('执行结果与日志')}</h2><p>{ent('客户产品里，任何自动化动作都应可追踪、可解释、可复盘。')}</p></div><button class="btn" id="refreshRuns">{ent('刷新日志')}</button></div>
<section class="list dense" id="skillRuns"></section>
<script>
async function loadSkills() {{
  const data = await apiFetch("/api/skills");
  const grid = qs("#skillGrid"); clearNode(grid);
  data.skills.forEach((skill) => grid.appendChild(el("div", {{ class: "card value-card" }}, [el("strong", {{ text: skill.title }}), el("p", {{ class: "muted", text: skill.description }}), el("div", {{ class: "muted", text: `${{skill.name}} · v${{skill.version}} · ${{(skill.permissions || []).join(", ")}}` }})])));
  const tools = await apiFetch("/api/mcp/tools");
  const mcp = qs("#mcpTools"); clearNode(mcp);
  const table = el("table"); table.appendChild(el("thead", {{}}, [el("tr", {{}}, ["工具", "说明", "权限"].map((text) => el("th", {{ text }})))]));
  const body = el("tbody");
  tools.tools.forEach((tool) => body.appendChild(el("tr", {{}}, [el("td", {{ text: tool.name }}), el("td", {{ text: tool.description }}), el("td", {{ text: (tool.annotations?.permissions || []).join(", ") }})])));
  table.appendChild(body); mcp.appendChild(table);
  loadRuns();
}}
async function loadRuns() {{
  const node = qs("#skillRuns"); clearNode(node);
  const data = await apiFetch("/api/skills/runs?limit=12");
  if (!data.runs.length) return node.appendChild(el("div", {{ class: "item muted", text: "暂无 Skill 运行日志" }}));
  data.runs.forEach((run) => node.appendChild(el("div", {{ class: "item compact" }}, [el("strong", {{ text: `${{run.run_id}} · ${{run.skill}} · ${{run.status}}` }}), el("div", {{ class: "muted", text: `${{run.started_at}} -> ${{run.finished_at}}` }})])));
}}
document.addEventListener("DOMContentLoaded", loadSkills);
document.addEventListener("DOMContentLoaded", () => qs("#refreshRuns").addEventListener("click", loadRuns));
</script>
"""


Path("templates/index.html").write_text(
    page("审脉 AuditPilot", "home", "审计交付总览", index_body, "行业落地能力：审计项目、证据请求、控制测试、底稿索引、质量门和整改闭环。"),
    encoding="utf-8",
)
Path("templates/skills.html").write_text(
    page("Skills / MCP 工具层", "skills", "Agent 能力中心", skills_body, "把 Agent 能力从代码函数升级为可治理的工具资产。"),
    encoding="utf-8",
)

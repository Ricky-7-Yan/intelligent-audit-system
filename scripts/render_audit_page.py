# -*- coding: utf-8 -*-
from pathlib import Path


def ent(text: str) -> str:
    return "".join(f"&#{ord(ch)};" if ord(ch) > 127 else ch for ch in text)


def write() -> None:
    body = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{ent('审计分析')} - {ent('智能审计 Agent 平台')}</title>
  <link rel="stylesheet" href="../static/app.css">
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><div class="brand-icon">&#23457;</div><span>{ent('智能审计')}</span></div>
      <nav class="nav-list">
        <a class="nav-link" href="/">&#8962; <span>{ent('总览')}</span></a>
        <a class="nav-link" href="/chat">&#9679; <span>{ent('审计对话')}</span></a>
        <a class="nav-link active" href="/audit">&#8981; <span>{ent('审计分析')}</span></a>
        <a class="nav-link" href="/knowledge">&#9638; <span>{ent('知识库')}</span></a>
        <a class="nav-link" href="/training">&#9633; <span>{ent('评估')}</span></a>
        <a class="nav-link" href="/skills">&#9671; <span>Skills</span></a>
      </nav>
      <div class="sidebar-note">{ent('从审计场景到证据、控制、质量门、报告和整改闭环。')}</div>
    </aside>
    <main class="main">
      <div class="topbar">
        <div class="page-title">
          <h1>{ent('审计分析')}</h1>
          <p>{ent('面向客户交付的审计指挥台：模板化启动、自动化分析、人工复核和整改跟踪。')}</p>
        </div>
        <div class="status-pill"><span class="status-dot"></span><span id="healthText">{ent('加载状态中')}</span></div>
      </div>

      <section class="command-panel">
        <div class="panel">
          <h2 class="panel-title">{ent('审计指挥栏')}</h2>
          <div class="grid grid-4">
            <div class="field"><label>{ent('审计对象')}</label><input id="auditItem" value="{ent('ERP系统权限管理')}"></div>
            <div class="field"><label>{ent('审计类型')}</label><select id="auditType"><option>{ent('安全审计')}</option><option>{ent('合规审计')}</option><option>{ent('风险评估')}</option><option>{ent('内部控制审计')}</option></select></div>
            <div class="field"><label>{ent('参考标准')}</label><select id="standardType"><option>ISO27001</option><option>COBIT</option><option>SOX</option><option>{ent('数据安全法')}</option></select></div>
            <div class="field"><label>{ent('关注风险')}</label><select id="riskLevel"><option>{ent('高')}</option><option>{ent('中')}</option><option>{ent('低')}</option></select></div>
          </div>
          <div class="toolbar mt-16"><button class="btn primary" id="runAudit">{ent('运行 Agent 审计')}</button><button class="btn" id="loadControls">{ent('查看控制库')}</button><a class="btn" id="downloadReport" href="#" target="_blank">{ent('下载报告')}</a></div>
          <div class="section-title"><div><h2>{ent('高频场景')}</h2><p>{ent('成熟产品要让客户少填表，直接从常见业务场景启动。')}</p></div></div>
          <div class="scenario-grid">
            <div class="card scenario" data-item="{ent('ERP系统权限管理')}" data-type="{ent('安全审计')}" data-standard="ISO27001" data-risk="{ent('高')}"><strong>ERP {ent('权限')}</strong><p class="muted">{ent('用户生命周期、职责分离、特权账号和权限复核。')}</p></div>
            <div class="card scenario" data-item="{ent('财务报告系统变更流程')}" data-type="{ent('内部控制审计')}" data-standard="SOX" data-risk="{ent('高')}"><strong>SOX ITGC</strong><p class="muted">{ent('变更审批、测试证据、上线记录和回退方案。')}</p></div>
            <div class="card scenario" data-item="{ent('客户敏感数据处理流程')}" data-type="{ent('合规审计')}" data-standard="{ent('数据安全法')}" data-risk="{ent('中')}"><strong>{ent('数据安全')}</strong><p class="muted">{ent('分类分级、授权、加密、脱敏、共享和日志审计。')}</p></div>
          </div>
        </div>
        <aside class="panel">
          <h2 class="panel-title">{ent('执行轨迹')}</h2>
          <div id="tracePanel" class="trace-list"><p class="muted">{ent('运行审计后展示 Agent 每一步的状态。')}</p></div>
        </aside>
      </section>

      <section class="kpi-row mt-16">
        <div class="card metric"><div class="metric-value" id="riskLevelCard">-</div><div class="metric-label">{ent('剩余风险')}</div></div>
        <div class="card metric"><div class="metric-value" id="riskScoreCard">-</div><div class="metric-label">{ent('风险评分')}</div></div>
        <div class="card metric"><div class="metric-value" id="complianceCard">-</div><div class="metric-label">{ent('合规评分')}</div></div>
        <div class="card metric"><div class="metric-value" id="qualityCard">-</div><div class="metric-label">{ent('质量门')}</div></div>
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
        <div class="panel"><h2 class="panel-title">{ent('审计档案')}</h2><div class="toolbar"><button class="btn" id="refreshRuns">{ent('刷新历史')}</button></div><div id="runHistory" class="list dense mt-16"><p class="muted">{ent('暂无历史')}</p></div></div>
        <div class="panel"><h2 class="panel-title">{ent('人工复核')}</h2><div class="grid grid-2"><div class="field"><label>{ent('复核人')}</label><input id="reviewer" value="{ent('审计经理')}"></div><div class="field"><label>{ent('结论')}</label><select id="reviewDecision"><option value="approve">{ent('通过')}</option><option value="need_evidence">{ent('补充证据')}</option><option value="reject">{ent('退回整改')}</option></select></div></div><div class="field mt-12"><label>{ent('复核意见')}</label><textarea id="reviewComment" rows="4" placeholder="{ent('记录人工判断、证据缺口或整改要求')}"></textarea></div><button class="btn primary mt-16" id="submitReview">{ent('提交复核')}</button><div id="reviewResult" class="muted mt-12"></div></div>
      </section>
    </main>
  </div>
  <script src="../static/app.js"></script>
  <script>
    let currentRunId = null;
    function statusBadge(status) {{ return el("span", {{ class: `badge ${{status}}`, text: status || "-" }}); }}
    function renderTrace(trace) {{ const node = qs("#tracePanel"); clearNode(node); (trace || []).forEach((item) => node.appendChild(el("div", {{ class: "trace-item" }}, [el("div", {{ class: "trace-stage", text: item.stage }}), el("div", {{}}, [el("strong", {{ text: item.status }}), el("div", {{ class: "muted", text: item.detail || "" }})])]))) ; if (!node.childElementCount) node.appendChild(el("p", {{ class: "muted", text: "暂无执行轨迹" }})); }}
    function renderTaskPlan(tasks) {{ const node = qs("#taskPlan"); clearNode(node); (tasks || []).forEach((task, index) => node.appendChild(el("div", {{ class: "timeline-step" }}, [el("div", {{ class: "step-index", text: String(index + 1) }}), el("div", {{ class: "item compact" }}, [el("strong", {{ text: task.name }}), el("div", {{ class: "muted", text: task.objective }}), el("div", {{ class: "mt-12", text: `负责人：${{task.owner}}` }})])]))); }}
    function renderEvidence(items) {{ const node = qs("#evidencePack"); clearNode(node); (items || []).forEach((item) => node.appendChild(el("div", {{ class: "item compact" }}, [el("strong", {{ text: `${{item.id}} · ${{item.source}}` }}), el("p", {{ class: "muted", text: item.summary }}), el("div", {{ text: `用途：${{item.usage}}` }})]))); }}
    function renderQuality(quality) {{ const node = qs("#qualityPanel"); clearNode(node); const percent = Math.round((quality.confidence || 0) * 100); const ring = el("div", {{ class: "quality-ring" }}, [el("span", {{ text: `${{percent}}%` }})]); ring.style.setProperty("--score", percent); node.appendChild(el("div", {{ class: "quality" }}, [ring, el("div", {{}}, [statusBadge(quality.status || "review"), el("p", {{ class: "muted", text: quality.review_note || "" }}), el("div", {{ text: `证据扎实度：${{quality.groundedness || 0}} · 控制覆盖：${{quality.control_coverage || 0}}` }}), el("div", {{ class: "mt-12 muted", text: `缺失证据：${{(quality.missing_evidence || []).join("、") || "无"}}` }})])]))); }}
    function renderMatrix(rows) {{ const node = qs("#controlMatrix"); clearNode(node); const table = el("table"); table.appendChild(el("thead", {{}}, [el("tr", {{}}, ["控制", "领域", "测试程序", "证据要求", "成熟度", "状态"].map((text) => el("th", {{ text }})))])); const body = el("tbody"); (rows || []).forEach((row) => body.appendChild(el("tr", {{}}, [el("td", {{ text: row.control_id || row.id }}), el("td", {{ text: row.domain }}), el("td", {{ text: row.test_procedure }}), el("td", {{ text: (row.evidence_required || []).join("、") }}), el("td", {{ text: String(row.maturity_level ?? "-") }}), el("td", {{ text: row.status || row.objective || "" }})]))); table.appendChild(body); node.appendChild(table); }}
    function renderAuditProgram(items) {{ const node = qs("#auditProgram"); clearNode(node); if (!(items || []).length) return node.appendChild(el("p", {{ class: "muted", text: "暂无审计程序" }})); items.forEach((item) => node.appendChild(el("div", {{ class: "item compact" }}, [el("strong", {{ text: `${{item.step_id}} · ${{item.control_id}}` }}), el("div", {{ class: "muted", text: item.procedure }}), el("div", {{ class: "mt-12", text: `认定：${{item.assertion}}` }}), el("div", {{ class: "muted", text: `底稿：${{item.workpaper_ref}}` }})]))); }}
    function renderSamplingPlan(plan) {{ const node = qs("#samplingPlan"); clearNode(node); if (!plan || !plan.population) return node.appendChild(el("p", {{ class: "muted", text: "暂无抽样计划" }})); [["总体", plan.population], ["期间", plan.period], ["方法", plan.method], ["样本量", plan.sample_size], ["分层", (plan.strata || []).join("、")], ["例外处理", plan.exception_handling]].forEach(([label, value]) => node.appendChild(el("div", {{ class: "item compact" }}, [el("strong", {{ text: label }}), el("div", {{ class: "muted", text: String(value || "") }})]))); }}
    function renderFindings(items) {{ const node = qs("#findings"); clearNode(node); if (!(items || []).length) return node.appendChild(el("p", {{ class: "muted", text: "当前未形成重大审计发现草稿" }})); items.forEach((finding) => node.appendChild(el("div", {{ class: "card" }}, [el("strong", {{ text: `${{finding.finding_id}} · ${{finding.title}}` }}), el("p", {{ class: "muted", text: `严重程度：${{finding.severity}}` }}), el("div", {{ text: finding.condition }}), el("p", {{ class: "muted", text: `影响：${{finding.effect}}` }}), el("div", {{ text: `建议：${{finding.recommendation}}` }})]))); }}
    function renderRecommendations(items) {{ const node = qs("#recommendations"); clearNode(node); (items || []).forEach((rec) => node.appendChild(el("div", {{ class: "card" }}, [el("strong", {{ text: `${{rec.type}} · ${{rec.priority}}` }}), el("p", {{ class: "muted", text: rec.description }}), el("div", {{ text: (rec.action_items || []).join("；") }}), el("div", {{ class: "mt-12 muted", text: `责任角色：${{rec.owner_role}} · ${{rec.due_days}}天 · ${{rec.success_metric}}` }})]))); }}
    function renderTasks(tasks) {{ const node = qs("#remediationTasks"); clearNode(node); if (!(tasks || []).length) return node.appendChild(el("p", {{ class: "muted", text: "暂无整改任务" }})); tasks.forEach((task) => {{ const status = el("select"); ["未开始", "进行中", "待验证", "已完成", "已关闭"].forEach((value) => {{ const option = el("option", {{ value, text: value }}); if (value === task.status) option.selected = true; status.appendChild(option); }}); const owner = el("input", {{ value: task.owner || "", placeholder: "责任人" }}); const note = el("input", {{ placeholder: "更新备注" }}); const save = el("button", {{ class: "btn", text: "更新" }}); save.addEventListener("click", () => updateTask(task.task_id, status.value, owner.value, note.value)); node.appendChild(el("div", {{ class: "item" }}, [el("strong", {{ text: `${{task.task_id}} · ${{task.title}} · ${{task.priority}}` }}), el("p", {{ class: "muted", text: task.description }}), el("div", {{ text: `验收指标：${{task.success_metric}}` }}), el("div", {{ class: "grid grid-4 mt-12" }}, [status, owner, note, save])])); }}); }}
    function renderResult(result) {{ setText("#auditResult", result.response); setText("#riskLevelCard", result.risk_assessment.risk_level); setText("#riskScoreCard", result.risk_assessment.risk_score); setText("#complianceCard", result.compliance_check.compliance_score); setText("#qualityCard", result.quality_gate.confidence); setText("#controlsCard", result.control_matrix.length); renderTrace(result.execution_trace || []); renderTaskPlan(result.task_plan || []); renderEvidence(result.evidence_pack || []); renderQuality(result.quality_gate || {{}}); renderMatrix(result.control_matrix || []); renderAuditProgram(result.audit_program || []); renderSamplingPlan(result.sampling_plan || {{}}); renderFindings(result.findings || []); renderRecommendations(result.recommendations || []); }}
    function renderRunRecord(record) {{ renderResult(record.result); renderTasks(record.remediation_tasks || []); }}
    async function loadRuns() {{ const node = qs("#runHistory"); clearNode(node); try {{ const data = await apiFetch("/api/audit/runs?limit=10"); if (!data.runs.length) return node.appendChild(el("p", {{ class: "muted", text: "暂无历史" }})); data.runs.forEach((run) => {{ const item = el("div", {{ class: "item compact clickable" }}, [el("strong", {{ text: `${{run.run_id}} · ${{run.audit_item || ""}}` }}), el("div", {{ class: "muted", text: `${{run.status}} · 风险 ${{run.risk_level || "-"}} · 合规 ${{run.compliance_score ?? "-"}} · 质量 ${{run.quality_confidence ?? "-"}}` }}), el("div", {{ class: "mt-12", text: run.created_at || "" }})]); item.addEventListener("click", () => loadRunDetail(run.run_id)); node.appendChild(item); }}); }} catch (error) {{ node.appendChild(el("p", {{ class: "muted", text: `加载失败：${{error.message}}` }})); }} }}
    async function loadRunDetail(runId) {{ const data = await apiFetch(`/api/audit/runs/${{encodeURIComponent(runId)}}`); currentRunId = runId; qs("#downloadReport").href = serviceUrl(`/api/audit/runs/${{encodeURIComponent(runId)}}/report.md`); renderRunRecord(data.run); }}
    async function runAudit() {{ setText("#auditResult", "正在运行 Agent 工作流..."); renderTrace([{{stage:"planner", status:"running", detail:"正在生成审计任务计划"}}]); try {{ const data = await apiFetch("/api/audit", {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify({{ audit_item: qs("#auditItem").value, audit_type: qs("#auditType").value, standard_type: qs("#standardType").value, risk_level: qs("#riskLevel").value }}) }}); currentRunId = data.run_id; qs("#downloadReport").href = serviceUrl(`/api/audit/runs/${{encodeURIComponent(currentRunId)}}/report.md`); renderResult(data.result); loadRunDetail(currentRunId); loadRuns(); }} catch (error) {{ setText("#auditResult", `分析失败：${{error.message}}`); }} }}
    qs("#runAudit").addEventListener("click", runAudit);
    qs("#loadControls").addEventListener("click", async () => {{ const data = await apiFetch("/api/audit/controls"); renderMatrix(data.controls); }});
    qs("#refreshRuns").addEventListener("click", loadRuns);
    qs("#submitReview").addEventListener("click", async () => {{ if (!currentRunId) return setText("#reviewResult", "请先运行或选择一条审计档案"); try {{ await apiFetch(`/api/audit/runs/${{encodeURIComponent(currentRunId)}}/review`, {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify({{ reviewer: qs("#reviewer").value, decision: qs("#reviewDecision").value, comment: qs("#reviewComment").value }}) }}); setText("#reviewResult", "复核已保存"); loadRuns(); }} catch (error) {{ setText("#reviewResult", `复核失败：${{error.message}}`); }} }});
    async function updateTask(taskId, status, owner, note) {{ if (!currentRunId) return; try {{ const data = await apiFetch(`/api/audit/runs/${{encodeURIComponent(currentRunId)}}/tasks/${{encodeURIComponent(taskId)}}`, {{ method: "POST", headers: {{ "Content-Type": "application/json" }}, body: JSON.stringify({{ status, owner, note }}) }}); renderRunRecord(data.run); loadRuns(); }} catch (error) {{ setText("#reviewResult", `任务更新失败：${{error.message}}`); }} }}
    qsa(".scenario").forEach((node) => node.addEventListener("click", () => {{ qs("#auditItem").value = node.dataset.item; qs("#auditType").value = node.dataset.type; qs("#standardType").value = node.dataset.standard; qs("#riskLevel").value = node.dataset.risk; }}));
    loadRuns();
  </script>
</body>
</html>
"""
    Path("templates/audit.html").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    write()

let currentRunId = null;
let currentEvidenceAnalysisId = null;

function normalizeRisk(value) {
  if (value === "高" || value === "high") return "high";
  if (value === "中" || value === "medium") return "medium";
  if (value === "低" || value === "low") return "low";
  return value || "medium";
}

function badge(status) {
  return el("span", { class: `badge ${status || ""}`, text: status || "-" });
}

function renderTrace(trace) {
  const node = qs("#tracePanel");
  clearNode(node);
  (trace || []).forEach((item) => {
    node.appendChild(el("div", { class: "trace-item" }, [
      el("div", { class: "trace-stage", text: item.stage }),
      el("div", {}, [el("strong", { text: item.status }), el("div", { class: "muted", text: item.detail || "" })]),
    ]));
  });
  if (!node.childElementCount) node.appendChild(el("p", { class: "muted", text: "暂无执行轨迹" }));
}

function renderTaskPlan(tasks) {
  const node = qs("#taskPlan");
  clearNode(node);
  (tasks || []).forEach((task, index) => {
    node.appendChild(el("div", { class: "timeline-step" }, [
      el("div", { class: "step-index", text: String(index + 1) }),
      el("div", { class: "item compact" }, [
        el("strong", { text: task.name }),
        el("div", { class: "muted", text: task.objective }),
        el("div", { class: "mt-12", text: `负责人：${task.owner}` }),
      ]),
    ]));
  });
}

function renderEvidence(items) {
  const node = qs("#evidencePack");
  clearNode(node);
  (items || []).forEach((item) => {
    node.appendChild(el("div", { class: "item compact" }, [
      el("strong", { text: `${item.id} · ${item.source}` }),
      el("p", { class: "muted", text: item.summary }),
      el("div", { text: `用途：${item.usage}` }),
    ]));
  });
}

function renderQuality(quality) {
  const node = qs("#qualityPanel");
  clearNode(node);
  const percent = Math.round((quality.confidence || 0) * 100);
  const ring = el("div", { class: "quality-ring" }, [el("span", { text: `${percent}%` })]);
  ring.style.setProperty("--score", percent);
  node.appendChild(el("div", { class: "quality" }, [
    ring,
    el("div", {}, [
      badge(quality.status || "review"),
      el("p", { class: "muted", text: quality.review_note || "" }),
      el("div", { text: `证据扎实度：${quality.groundedness || 0} · 控制覆盖：${quality.control_coverage || 0}` }),
      el("div", { class: "mt-12 muted", text: `缺失证据：${(quality.missing_evidence || []).join("、") || "无"}` }),
    ]),
  ]));
}

function renderMatrix(rows) {
  const node = qs("#controlMatrix");
  clearNode(node);
  const table = el("table");
  table.appendChild(el("thead", {}, [el("tr", {}, ["控制", "领域", "测试程序", "证据要求", "成熟度", "状态"].map((text) => el("th", { text })))]));
  const body = el("tbody");
  (rows || []).forEach((row) => {
    body.appendChild(el("tr", {}, [
      el("td", { text: row.control_id || row.id }),
      el("td", { text: row.domain }),
      el("td", { text: row.test_procedure || row.objective || "" }),
      el("td", { text: (row.evidence_required || []).join("、") }),
      el("td", { text: String(row.maturity_level ?? "-") }),
      el("td", { text: row.status || "" }),
    ]));
  });
  table.appendChild(body);
  node.appendChild(table);
}

function renderListCard(node, items, emptyText, build) {
  clearNode(node);
  if (!(items || []).length) {
    node.appendChild(el("p", { class: "muted", text: emptyText }));
    return;
  }
  items.forEach((item) => node.appendChild(build(item)));
}

function renderAuditProgram(items) {
  renderListCard(qs("#auditProgram"), items, "暂无审计程序", (item) => el("div", { class: "item compact" }, [
    el("strong", { text: `${item.step_id} · ${item.control_id}` }),
    el("div", { class: "muted", text: item.procedure }),
    el("div", { class: "mt-12", text: `认定：${item.assertion}` }),
    el("div", { class: "muted", text: `底稿：${item.workpaper_ref}` }),
  ]));
}

function renderSamplingPlan(plan) {
  const node = qs("#samplingPlan");
  clearNode(node);
  if (!plan || !plan.population) {
    node.appendChild(el("p", { class: "muted", text: "暂无抽样计划" }));
    return;
  }
  [["总体", plan.population], ["期间", plan.period], ["方法", plan.method], ["样本量", plan.sample_size], ["分层", (plan.strata || []).join("、")], ["例外处理", plan.exception_handling]].forEach(([label, value]) => {
    node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: label }), el("div", { class: "muted", text: String(value || "") })]));
  });
}

function renderFindings(items) {
  renderListCard(qs("#findings"), items, "当前未形成重大审计发现草稿", (finding) => el("div", { class: "card" }, [
    el("strong", { text: `${finding.finding_id} · ${finding.title}` }),
    el("p", { class: "muted", text: `严重程度：${finding.severity}` }),
    el("div", { text: finding.condition }),
    el("p", { class: "muted", text: `影响：${finding.effect || ""}` }),
    el("div", { text: `建议：${finding.recommendation}` }),
  ]));
}

function renderRecommendations(items) {
  renderListCard(qs("#recommendations"), items, "暂无整改建议", (rec) => el("div", { class: "card" }, [
    el("strong", { text: `${rec.type} · ${rec.priority}` }),
    el("p", { class: "muted", text: rec.description }),
    el("div", { text: (rec.action_items || []).join("；") }),
    el("div", { class: "mt-12 muted", text: `责任角色：${rec.owner_role} · ${rec.due_days}天 · ${rec.success_metric}` }),
  ]));
}

function renderTasks(tasks) {
  renderListCard(qs("#remediationTasks"), tasks, "暂无整改任务", (task) => {
    const status = el("select");
    ["未开始", "进行中", "待验证", "已完成", "已关闭"].forEach((value) => {
      const option = el("option", { value, text: value });
      if (value === task.status) option.selected = true;
      status.appendChild(option);
    });
    const owner = el("input", { value: task.owner || "", placeholder: "责任人" });
    const note = el("input", { placeholder: "更新备注" });
    const save = el("button", { class: "btn", text: "更新" });
    save.addEventListener("click", () => updateTask(task.task_id, status.value, owner.value, note.value));
    return el("div", { class: "item" }, [
      el("strong", { text: `${task.task_id} · ${task.title} · ${task.priority}` }),
      el("p", { class: "muted", text: task.description }),
      el("div", { text: `验收指标：${task.success_metric}` }),
      el("div", { class: "grid grid-4 mt-12" }, [status, owner, note, save]),
    ]);
  });
}

function renderEvidenceRequests(items) {
  renderListCard(qs("#evidenceRequests"), items, "暂无证据请求", (item) => {
    const status = el("select");
    ["待收集", "已收到", "需补充", "已验证", "不适用"].forEach((value) => {
      const option = el("option", { value, text: value });
      if (value === item.status) option.selected = true;
      status.appendChild(option);
    });
    const owner = el("input", { value: item.owner || "", placeholder: "责任人" });
    const note = el("input", { placeholder: "证据备注" });
    const save = el("button", { class: "btn", text: "保存" });
    save.addEventListener("click", () => updateEvidence(item.request_id, status.value, owner.value, note.value));
    return el("div", { class: "item compact" }, [
      el("strong", { text: `${item.request_id} · ${item.evidence}` }),
      el("div", { class: "muted", text: `${item.usage || ""} · 优先级：${item.priority || "中"}` }),
      el("div", { class: "grid grid-4 mt-12" }, [status, owner, note, save]),
    ]);
  });
}

function renderControlTests(items) {
  renderListCard(qs("#controlTests"), items, "暂无控制测试", (item) => {
    const result = el("select");
    ["待执行", "通过", "例外", "不适用", "需扩大样本"].forEach((value) => {
      const option = el("option", { value, text: value });
      if (value === item.result) option.selected = true;
      result.appendChild(option);
    });
    const tester = el("input", { value: item.tester || "", placeholder: "测试人" });
    const exception = el("input", { placeholder: "例外说明" });
    const save = el("button", { class: "btn", text: "保存" });
    save.addEventListener("click", () => updateControlTest(item.control_id, result.value, tester.value, exception.value));
    return el("div", { class: "item compact" }, [
      el("strong", { text: `${item.control_id} · ${item.domain} · ${item.workpaper_ref || ""}` }),
      el("div", { class: "muted", text: item.procedure || item.test_procedure || "" }),
      el("div", { class: "grid grid-4 mt-12" }, [result, tester, exception, save]),
    ]);
  });
}

function renderResult(result) {
  setText("#auditResult", result.response);
  setText("#riskLevelCard", result.risk_assessment.risk_level);
  setText("#riskScoreCard", result.risk_assessment.risk_score);
  setText("#complianceCard", result.compliance_check.compliance_score);
  setText("#qualityCard", result.quality_gate.confidence);
  renderTrace(result.execution_trace || []);
  renderTaskPlan(result.task_plan || []);
  renderEvidence(result.evidence_pack || []);
  renderQuality(result.quality_gate || {});
  renderMatrix(result.control_matrix || []);
  renderAuditProgram(result.audit_program || []);
  renderSamplingPlan(result.sampling_plan || {});
  renderFindings(result.findings || []);
  renderRecommendations(result.recommendations || []);
}

function renderRunRecord(record) {
  renderResult(record.result);
  renderTasks(record.remediation_tasks || []);
  renderEvidenceRequests(record.evidence_requests || []);
  renderControlTests(record.control_tests || []);
  setText("#stageCard", record.lifecycle_stage || "-");
}

function renderDeliveryPreview(pkg) {
  const node = qs("#deliveryPreview");
  clearNode(node);
  const cards = [["底稿索引", pkg.workpaper_index?.length || 0], ["证据请求", pkg.evidence_request_list?.length || 0], ["控制测试", pkg.control_test_plan?.length || 0], ["访谈计划", pkg.interview_plan?.length || 0], ["现场日程", pkg.fieldwork_calendar?.length || 0], ["发现跟踪", pkg.finding_tracker?.length || 0], ["复核要求", pkg.signoff?.review_required ? "是" : "否"], ["项目阶段", pkg.engagement?.lifecycle_stage || "-"]];
  cards.forEach(([label, value]) => node.appendChild(el("div", { class: "card delivery-card" }, [el("div", { class: "delivery-number", text: String(value) }), el("div", { class: "metric-label", text: label })])));
  if (pkg.evidence_analysis_index && pkg.evidence_analysis_index.length) {
    node.appendChild(el("div", { class: "card delivery-card" }, [
      el("div", { class: "delivery-number", text: String(pkg.evidence_analysis_index.length) }),
      el("div", { class: "metric-label", text: "证据分析底稿" }),
    ]));
  }
}

function renderEvidenceAnalysis(analysis) {
  const node = qs("#evidenceAnalysisResult");
  clearNode(node);
  currentEvidenceAnalysisId = analysis.analysis_id;
  const profile = analysis.profile || {};
  const gate = analysis.quality_gate || {};
  node.appendChild(el("div", { class: "grid grid-4" }, [
    el("div", { class: "card metric" }, [el("div", { class: "metric-value small", text: String(profile.rows || 0) }), el("div", { class: "metric-label", text: "记录行数" })]),
    el("div", { class: "card metric" }, [el("div", { class: "metric-value small", text: String(profile.field_count || 0) }), el("div", { class: "metric-label", text: "字段数量" })]),
    el("div", { class: "card metric" }, [el("div", { class: "metric-value small", text: String((analysis.risk_signals || []).length) }), el("div", { class: "metric-label", text: "风险信号" })]),
    el("div", { class: "card metric" }, [el("div", { class: "metric-value small", text: String(gate.confidence || 0) }), el("div", { class: "metric-label", text: gate.status || "review" })]),
  ]));
  node.appendChild(el("div", { class: "item compact" }, [
    el("div", { class: "item-head" }, [
      el("strong", { text: `字段画像 · ${analysis.analysis_id}` }),
      el("button", { class: "btn", onclick: attachCurrentEvidenceAnalysis, text: "归档到当前项目" }),
    ]),
    el("p", { class: "muted", text: (profile.fields || []).slice(0, 24).join("、") || "未识别到结构化字段" }),
  ]));
  renderListCard(node, analysis.risk_signals || [], "未发现明显风险信号", (signal) => el("div", { class: "item compact" }, [
    el("strong", { text: `${signal.title} · ${signal.severity}` }),
    el("div", { class: "muted", text: `影响行数 ${signal.affected_rows || 0} · 置信度 ${signal.confidence}` }),
    el("p", { class: "muted", text: signal.audit_implication }),
  ]));
  renderListCard(node, analysis.mapped_controls || [], "暂无控制映射", (control) => el("div", { class: "item compact" }, [
    el("strong", { text: `${control.control_id} · ${control.domain}` }),
    el("p", { class: "muted", text: control.test_procedure }),
    el("div", { text: `命中关键词：${(control.hit_keywords || []).join("、")}` }),
  ]));
  renderListCard(node, analysis.evidence_requests || [], "暂无补证建议", (request) => el("div", { class: "item compact" }, [
    el("strong", { text: `${request.evidence} · ${request.priority}` }),
    el("p", { class: "muted", text: request.usage }),
    el("div", { text: `建议字段：${(request.required_fields || []).join("、")}` }),
  ]));
  (analysis.recommended_next_steps || []).forEach((step) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: "下一步" }), el("p", { class: "muted", text: step })])));
}

function renderEvidenceAnalysisHistory(items) {
  const node = qs("#evidenceAnalysisHistory");
  clearNode(node);
  if (!(items || []).length) {
    node.appendChild(el("p", { class: "muted", text: "暂无证据分析记录。" }));
    return;
  }
  items.forEach((item) => {
    const card = el("div", { class: "item compact clickable" }, [
      el("strong", { text: `${item.analysis_id} · ${item.file_name}` }),
      el("div", { class: "muted", text: `风险 ${item.risk_count || 0} · 控制 ${item.control_count || 0} · ${item.quality_gate?.label || item.quality_gate?.status || "review"}` }),
      el("div", { class: "muted", text: item.created_at || "" }),
    ]);
    card.addEventListener("click", () => loadEvidenceAnalysis(item.analysis_id));
    node.appendChild(card);
  });
}

async function loadEvidenceAnalyses() {
  try {
    const data = await apiFetch("/api/evidence/analyses?limit=10");
    renderEvidenceAnalysisHistory(data.analyses || []);
  } catch (error) {
    const node = qs("#evidenceAnalysisHistory");
    clearNode(node);
    node.appendChild(el("p", { class: "muted", text: `证据分析历史加载失败：${error.message}` }));
  }
}

async function loadEvidenceAnalysis(analysisId) {
  const data = await apiFetch(`/api/evidence/analyses/${encodeURIComponent(analysisId)}`);
  renderEvidenceAnalysis(data.analysis);
}

async function analyzeEvidenceFile() {
  const input = qs("#evidenceFile");
  const file = input.files && input.files[0];
  const node = qs("#evidenceAnalysisResult");
  clearNode(node);
  if (!file) {
    node.appendChild(el("p", { class: "muted", text: "请先选择证据文件。" }));
    return;
  }
  node.appendChild(el("p", { class: "muted", text: "正在分析证据文件..." }));
  const form = new FormData();
  form.append("file", file);
  form.append("audit_item", qs("#auditItem").value || "");
  form.append("audit_type", qs("#evidenceAnalysisType").value || qs("#auditType").value || "");
  form.append("standard_type", qs("#standardType").value || "");
  try {
    const data = await fetch(apiUrl("/api/evidence/analyze"), { method: "POST", body: form }).then(async (response) => {
      const payload = await response.json().catch(() => ({}));
      if (!response.ok || payload.success === false) throw new Error(payload.detail || `请求失败: ${response.status}`);
      return payload;
    });
    renderEvidenceAnalysis(data.analysis);
    loadEvidenceAnalyses();
  } catch (error) {
    clearNode(node);
    node.appendChild(el("p", { class: "muted", text: `证据分析失败：${error.message}` }));
  }
}

async function attachCurrentEvidenceAnalysis() {
  const node = qs("#evidenceAnalysisResult");
  if (!currentRunId) {
    node.appendChild(el("p", { class: "muted", text: "请先运行或选择一个审计项目，再归档证据分析。" }));
    return;
  }
  if (!currentEvidenceAnalysisId) {
    node.appendChild(el("p", { class: "muted", text: "请先完成一次证据分析。" }));
    return;
  }
  try {
    const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(currentRunId)}/evidence-analyses`, {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ analysis_id: currentEvidenceAnalysisId }),
    });
    renderRunRecord(data.run);
    loadRuns();
    loadDelivery(currentRunId);
    node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: "已归档" }), el("p", { class: "muted", text: "证据分析已进入当前项目的底稿索引、补证清单和控制测试工作台。" })]));
  } catch (error) {
    node.appendChild(el("p", { class: "muted", text: `归档失败：${error.message}` }));
  }
}

async function loadDelivery(runId) {
  try {
    const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(runId)}/delivery`);
    renderDeliveryPreview(data.package);
  } catch (error) {
    setText("#reviewResult", `交付包加载失败：${error.message}`);
  }
}

async function loadRuns() {
  const node = qs("#runHistory");
  clearNode(node);
  try {
    const data = await apiFetch("/api/audit/runs?limit=10");
    if (!data.runs.length) {
      node.appendChild(el("p", { class: "muted", text: "暂无历史" }));
      return;
    }
    data.runs.forEach((run) => {
      const item = el("div", { class: "item compact clickable" }, [
        el("strong", { text: `${run.run_id} · ${run.audit_item || ""}` }),
        el("div", { class: "muted", text: `${run.lifecycle_stage || "-"} · ${run.status} · 风险 ${run.risk_level || "-"} · 合规 ${run.compliance_score ?? "-"} · 质量 ${run.quality_confidence ?? "-"}` }),
        el("div", { class: "mt-12", text: run.created_at || "" }),
      ]);
      item.addEventListener("click", () => loadRunDetail(run.run_id));
      node.appendChild(item);
    });
  } catch (error) {
    node.appendChild(el("p", { class: "muted", text: `加载失败：${error.message}` }));
  }
}

async function loadRunDetail(runId) {
  const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(runId)}`);
  currentRunId = runId;
  qs("#downloadReport").href = serviceUrl(`/api/audit/runs/${encodeURIComponent(runId)}/report.md`);
  qs("#downloadDelivery").href = serviceUrl(`/api/audit/runs/${encodeURIComponent(runId)}/delivery.md`);
  renderRunRecord(data.run);
  loadDelivery(runId);
}

async function runAudit() {
  const button = qs("#runAudit");
  button.disabled = true;
  button.textContent = "运行中...";
  setText("#auditResult", "正在运行 Agent 工作流...");
  renderTrace([{ stage: "planner", status: "running", detail: "正在生成审计任务计划" }]);
  try {
    const data = await apiFetch("/api/audit", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({
        audit_item: qs("#auditItem").value,
        audit_type: qs("#auditType").value,
        standard_type: qs("#standardType").value,
        risk_level: qs("#riskLevel").value,
        business_context: qs("#businessContext").value,
        audit_scope: qs("#auditScope").value,
        audit_period: qs("#auditPeriod").value,
        key_questions: qs("#keyQuestions").value,
        existing_evidence: qs("#existingEvidence").value,
      }),
    });
    currentRunId = data.run_id;
    qs("#downloadReport").href = serviceUrl(`/api/audit/runs/${encodeURIComponent(currentRunId)}/report.md`);
    qs("#downloadDelivery").href = serviceUrl(`/api/audit/runs/${encodeURIComponent(currentRunId)}/delivery.md`);
    renderRunRecord(data.run);
    loadRuns();
    loadDelivery(currentRunId);
  } catch (error) {
    setText("#auditResult", `分析失败：${error.message}`);
  } finally {
    button.disabled = false;
    button.textContent = "运行 Agent 审计";
  }
}

async function runResearch() {
  const node = qs("#researchPanel");
  clearNode(node);
  node.appendChild(el("div", { class: "item compact muted", text: "正在执行多路查询、来源融合和推理验证..." }));
  try {
    const question = `${qs("#auditItem").value} 如何开展 ${qs("#auditType").value}，需要哪些证据、控制测试和复核条件？`;
    const data = await apiFetch("/api/research/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ question, context: { standard_type: qs("#standardType").value } }),
    });
    const result = data.result;
    clearNode(node);
    node.appendChild(el("div", { class: "item" }, [
      el("strong", { text: "研究结论" }),
      el("p", { class: "prewrap muted", text: result.answer }),
    ]));
    node.appendChild(el("div", { class: "grid grid-3" }, [
      el("div", { class: "card metric" }, [el("div", { class: "metric-value small", text: String(result.query_rewrites.length) }), el("div", { class: "metric-label", text: "查询改写" })]),
      el("div", { class: "card metric" }, [el("div", { class: "metric-value small", text: String(result.sources.length) }), el("div", { class: "metric-label", text: "融合来源" })]),
      el("div", { class: "card metric" }, [el("div", { class: "metric-value small", text: String(result.evaluation.faithfulness) }), el("div", { class: "metric-label", text: "真实性评分" })]),
    ]));
    (result.reasoning_trace || []).forEach((step) => {
      node.appendChild(el("div", { class: "item compact" }, [
        el("strong", { text: `${step.stage} · ${step.action}` }),
        el("div", { class: "muted", text: step.output }),
      ]));
    });
  } catch (error) {
    clearNode(node);
    node.appendChild(el("div", { class: "item muted", text: `Deep Research 失败：${error.message}` }));
  }
}

async function updateTask(taskId, status, owner, note) {
  if (!currentRunId) return;
  try {
    const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(currentRunId)}/tasks/${encodeURIComponent(taskId)}`, { method: "POST", headers: { "Content-Type": "application/json; charset=utf-8" }, body: JSON.stringify({ status, owner, note }) });
    renderRunRecord(data.run);
    loadRuns();
    loadDelivery(currentRunId);
  } catch (error) {
    setText("#reviewResult", `任务更新失败：${error.message}`);
  }
}

async function updateEvidence(requestId, status, owner, note) {
  if (!currentRunId) return;
  try {
    const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(currentRunId)}/evidence/${encodeURIComponent(requestId)}`, { method: "POST", headers: { "Content-Type": "application/json; charset=utf-8" }, body: JSON.stringify({ status, owner, note }) });
    renderRunRecord(data.run);
    loadRuns();
    loadDelivery(currentRunId);
  } catch (error) {
    setText("#reviewResult", `证据更新失败：${error.message}`);
  }
}

async function updateControlTest(controlId, result, tester, exception) {
  if (!currentRunId) return;
  try {
    const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(currentRunId)}/controls/${encodeURIComponent(controlId)}/test`, { method: "POST", headers: { "Content-Type": "application/json; charset=utf-8" }, body: JSON.stringify({ result, tester, exception }) });
    renderRunRecord(data.run);
    loadRuns();
    loadDelivery(currentRunId);
  } catch (error) {
    setText("#reviewResult", `控制测试更新失败：${error.message}`);
  }
}

function bindScenarios() {
  qsa(".scenario").forEach((node) => node.addEventListener("click", () => {
    qs("#auditItem").value = node.dataset.item;
    qs("#auditType").value = node.dataset.type;
    qs("#standardType").value = node.dataset.standard;
    qs("#riskLevel").value = normalizeRisk(node.dataset.risk);
    qs("#auditScope").value = node.dataset.scope || "";
    qs("#existingEvidence").value = node.dataset.evidence || "";
    qs("#keyQuestions").value = node.dataset.questions || "";
  }));
}

async function loadTemplates() {
  try {
    const data = await apiFetch("/api/audit/templates");
    const grid = qs("#scenarioGrid");
    clearNode(grid);
    data.templates.forEach((tpl) => {
      grid.appendChild(el("div", {
        class: "card scenario",
        "data-item": tpl.name,
        "data-type": tpl.audit_type,
        "data-standard": tpl.standard,
        "data-risk": tpl.risk_level,
        "data-scope": (tpl.scope || []).join("、"),
        "data-evidence": (tpl.evidence || []).slice(0, 8).join("、"),
        "data-questions": `${tpl.name} 的关键控制是否设计有效、运行证据是否充分、例外是否闭环？`,
      }, [
        el("strong", { text: tpl.name }),
        el("p", { class: "muted", text: `范围：${(tpl.scope || []).slice(0, 4).join("、")}` }),
        el("div", { class: "muted", text: `交付物：${(tpl.deliverables || []).slice(0, 2).join("、")}` }),
      ]));
    });
    bindScenarios();
  } catch {
    bindScenarios();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  qs("#runAudit").addEventListener("click", runAudit);
  qs("#loadControls").addEventListener("click", async () => {
    const data = await apiFetch("/api/audit/controls");
    renderMatrix(data.controls);
  });
  qs("#runResearch").addEventListener("click", runResearch);
  qs("#refreshRuns").addEventListener("click", loadRuns);
  qs("#analyzeEvidence").addEventListener("click", analyzeEvidenceFile);
  qs("#refreshEvidenceAnalyses").addEventListener("click", loadEvidenceAnalyses);
  qs("#submitReview").addEventListener("click", async () => {
    if (!currentRunId) {
      setText("#reviewResult", "请先运行或选择一条审计档案");
      return;
    }
    try {
      const data = await apiFetch(`/api/audit/runs/${encodeURIComponent(currentRunId)}/review`, { method: "POST", headers: { "Content-Type": "application/json; charset=utf-8" }, body: JSON.stringify({ reviewer: qs("#reviewer").value, decision: qs("#reviewDecision").value, comment: qs("#reviewComment").value }) });
      setText("#reviewResult", "复核已保存");
      renderRunRecord(data.run);
      loadRuns();
      loadDelivery(currentRunId);
    } catch (error) {
      setText("#reviewResult", `复核失败：${error.message}`);
    }
  });
  qsa(".seg").forEach((btn) => btn.addEventListener("click", () => {
    qsa(".seg").forEach((node) => node.classList.remove("active"));
    btn.classList.add("active");
  }));
  loadTemplates();
  loadRuns();
  loadEvidenceAnalyses();
});

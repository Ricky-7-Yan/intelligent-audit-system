const metricDefinitions = [
  ["faithfulness", "真实性", "答案能回溯到来源、证据或审计底稿"],
  ["completeness", "完整性", "覆盖范围、风险、证据、测试和整改"],
  ["audit_professionalism", "审计专业性", "使用审计术语、底稿、抽样和复核语言"],
  ["actionability", "可执行性", "输出明确动作、责任、验证和关闭标准"],
  ["compliance_alignment", "合规对齐", "覆盖 SOX/ISO/COBIT/制度要求"],
  ["agentic_capability", "Agentic 能力", "规划、工具、Memory、质量门和闭环"],
  ["tool_trace_quality", "轨迹质量", "规划、检索、映射、风险和质量门完整"],
  ["human_review_awareness", "人工复核", "能识别证据不足和升级复核"],
];

let customCases = [];
let activeMode = "agent";

function scoreText(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return Number(value).toFixed(3);
}

function percentText(value) {
  if (value === undefined || value === null || Number.isNaN(Number(value))) return "-";
  return `${Math.round(Number(value) * 100)}%`;
}

function setBusy(message) {
  const node = qs("#evalResults");
  clearNode(node);
  node.appendChild(el("div", { class: "item compact" }, [
    el("strong", { text: message }),
    el("div", { class: "skeleton-lines mt-12" }, [
      el("div", { class: "skeleton-line" }),
      el("div", { class: "skeleton-line", style: "width:86%" }),
      el("div", { class: "skeleton-line", style: "width:68%" }),
    ]),
  ]));
}

function setQuality(score) {
  const normalized = Math.max(0, Math.min(100, Math.round(Number(score || 0) * 100)));
  qs("#qualityRing").style.setProperty("--score", normalized);
  setText("#overallScore", scoreText(score));
}

function selectedMetrics() {
  return qsa("input[name='metric']:checked").map((item) => item.value);
}

function parseTerms(text) {
  return text
    .split(/[,，、\n]/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function currentCase() {
  return {
    id: qs("#caseId").value.trim() || "CUSTOM-001",
    category: qs("#caseCategory").value,
    question: qs("#caseQuestion").value.trim(),
    expected_answer: qs("#expectedAnswer").value.trim(),
    expected_terms: parseTerms(qs("#expectedTerms").value),
    evaluation_criteria: selectedMetrics(),
  };
}

function renderMetricToggles() {
  const node = qs("#metricToggles");
  clearNode(node);
  metricDefinitions.forEach(([key, name, desc]) => {
    const input = el("input", { type: "checkbox", name: "metric", value: key, checked: "checked" });
    const item = el("label", { class: "metric-toggle" }, [
      input,
      el("span", { class: "metric-toggle-text" }, [
        el("strong", { text: name }),
        el("small", { text: desc }),
      ]),
    ]);
    node.appendChild(item);
  });
}

function renderCustomCases() {
  const node = qs("#customCases");
  clearNode(node);
  if (!customCases.length) {
    node.appendChild(el("p", { class: "muted", text: "暂无自定义用例。" }));
    return;
  }
  customCases.forEach((item, index) => {
    node.appendChild(el("div", { class: "item compact" }, [
      el("strong", { text: item.id }),
      el("div", { class: "muted", text: `${item.category} · ${item.expected_terms.length} 个关键点` }),
      el("p", { class: "muted", text: item.question }),
      el("button", { class: "btn", onclick: () => removeCase(index), text: "移除" }),
    ]));
  });
}

function removeCase(index) {
  customCases = customCases.filter((_, current) => current !== index);
  renderCustomCases();
}

function renderMetrics(metrics = {}) {
  setQuality(metrics.overall_score || 0);
  setText("#totalTests", metrics.total_tests ?? "-");
  setText("#passRate", percentText(metrics.pass_rate));
  setText("#regressionCount", metrics.regression_count ?? "-");
  setText("#latencyScore", metrics.avg_latency_ms ? `${metrics.avg_latency_ms}ms` : "-");
  const metricScores = metrics.metric_scores || {};
  setText("#faithfulnessScore", scoreText(metricScores.faithfulness));
  const toolScore = metricScores.tool_trace_quality ?? metricScores.agentic_capability;
  setText("#toolScore", scoreText(toolScore));
}

function renderScorePills(evaluation = {}) {
  return el("div", { class: "score-grid" }, Object.entries(evaluation).map(([key, value]) =>
    el("div", { class: "score-pill" }, [
      el("span", { text: key }),
      el("strong", { text: scoreText(value) }),
    ])
  ));
}

function markdownBlock(text, className = "muted") {
  const node = el("div", { class: className });
  setMarkdown(node, text || "");
  return node;
}

function renderEvalResults(results) {
  const node = qs("#evalResults");
  clearNode(node);
  renderMetrics(results.overall_metrics || {});
  Object.entries(results.overall_metrics?.category_scores || {}).forEach(([category, score]) => {
    node.appendChild(el("div", { class: "item compact" }, [
      el("strong", { text: category }),
      el("div", { class: "muted", text: `分类得分：${scoreText(score)}` }),
    ]));
  });
  (results.results || []).forEach((item, index) => {
    const risks = item.regression_risks || [];
    const suggestions = item.optimization_suggestions || [];
    node.appendChild(el("details", { class: "item eval-card eval-detail", open: index === 0 }, [
      el("summary", {}, [
        el("strong", { text: `${item.test_id} · ${item.category}` }),
        el("span", { class: "badge", text: `${item.latency_ms || 0}ms` }),
      ]),
      el("p", { class: "muted", text: item.question }),
      renderScorePills(item.evaluation || {}),
      el("div", { class: "trace-summary" }, [
        el("span", { text: `轨迹步骤 ${item.trajectory?.steps ?? 0}` }),
        el("span", { text: `阶段覆盖 ${percentText(item.trajectory?.stage_coverage ?? 0)}` }),
        el("span", { text: `证据缺口 ${item.trajectory?.missing_evidence_count ?? 0}` }),
      ]),
      markdownBlock(item.actual_answer || ""),
      el("div", { class: "list dense" }, [
        el("div", { class: "item compact" }, [el("strong", { text: "回归风险" }), el("p", { class: "muted", text: risks.join("；") })]),
        el("div", { class: "item compact" }, [el("strong", { text: "优化建议" }), el("p", { class: "muted", text: suggestions.join("；") })]),
      ]),
    ]));
  });
}

function renderRagResults(results) {
  const node = qs("#evalResults");
  clearNode(node);
  setQuality(results.overall_score || 0);
  setText("#totalTests", results.total_cases ?? "-");
  setText("#passRate", "-");
  setText("#regressionCount", (results.results || []).reduce((sum, item) => sum + (item.failure_modes || []).filter((mode) => !mode.includes("未发现")).length, 0));
  setText("#faithfulnessScore", "-");
  setText("#toolScore", "-");
  setText("#authorityScore", scoreText(avg((results.results || []).map((item) => item.authority_score))));
  setText("#latencyScore", "-");
  (results.results || []).forEach((item, index) => {
    node.appendChild(el("details", { class: "item eval-card eval-detail", open: index === 0 }, [
      el("summary", {}, [
        el("strong", { text: `${item.case_id} · ${item.category}` }),
        el("span", { class: "badge", text: `${item.retrieved_docs_count || 0} sources` }),
      ]),
      el("p", { class: "muted", text: item.question }),
      renderScorePills({
        term_score: item.term_score,
        source_score: item.source_score,
        authority_score: item.authority_score,
        retrieval_confidence: item.retrieval_confidence,
        overall: item.overall,
      }),
      el("p", { class: "muted", text: `失败模式：${(item.failure_modes || []).join("；")}` }),
    ]));
  });
  (results.closed_loop_suggestions || []).forEach((text) => node.appendChild(el("div", { class: "item compact" }, [el("strong", { text: "闭环建议" }), el("p", { class: "muted", text })])));
}

function renderRunGate(run) {
  const gate = run.release_gate || {};
  const status = gate.status || "review";
  const cls = status === "pass" ? "pass" : status === "blocked" ? "blocked" : "review";
  return el("span", { class: `badge ${cls}`, text: gate.label || "需复核" });
}

function renderEvaluationRuns(runs) {
  const node = qs("#evaluationRuns");
  clearNode(node);
  if (!runs.length) {
    node.appendChild(el("p", { class: "muted", text: "暂无评测记录。" }));
    return;
  }
  runs.forEach((run) => {
    const metrics = run.metrics || {};
    const gate = run.release_gate || {};
    const deleteButton = el("button", { class: "btn danger ghost", type: "button", text: "删除" });
    deleteButton.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      deleteEvaluationRun(run.run_id);
    });
    node.appendChild(el("details", { class: "item eval-card eval-detail record-detail" }, [
      el("summary", {}, [
        el("div", { class: "record-title" }, [
          el("strong", { text: `${run.run_id} · ${run.run_type}` }),
          el("small", { text: run.created_at || "" }),
        ]),
        el("div", { class: "record-actions" }, [
          renderRunGate(run),
          deleteButton,
        ]),
      ]),
      el("div", { class: "detail-body" }, [
        el("div", { class: "trace-summary" }, [
          el("span", { text: `得分 ${scoreText(metrics.overall_score)}` }),
          el("span", { text: `通过率 ${percentText(metrics.pass_rate)}` }),
          el("span", { text: `回归 ${metrics.regression_count ?? 0}` }),
          el("span", { text: `基线 ${run.comparison?.baseline_run_id || "新基线"}` }),
        ]),
        el("p", { class: "muted", text: (gate.blockers || []).join("；") || "满足当前发布门禁。" }),
      ]),
    ]));
  });
}

async function deleteEvaluationRun(runId) {
  if (!runId) return;
  if (!window.confirm(`确认删除评测记录 ${runId}？`)) return;
  try {
    await apiFetch(`/api/evaluation/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
    await loadEvaluationRuns();
    showToast(`已删除评测记录 ${runId}`, "success");
  } catch (error) {
    showToast(`删除评测记录失败：${error.message}`, "error");
  }
}

function avg(values) {
  const clean = values.filter((item) => item !== undefined && item !== null);
  return clean.length ? clean.reduce((sum, item) => sum + Number(item), 0) / clean.length : null;
}

function renderResearch(result) {
  const node = qs("#evalResults");
  clearNode(node);
  setQuality(result.evaluation?.faithfulness || 0);
  setText("#totalTests", result.query_rewrites?.length || 0);
  setText("#passRate", result.evaluation?.requires_human_review ? "需复核" : "通过");
  setText("#regressionCount", result.evaluation?.requires_human_review ? 1 : 0);
  setText("#faithfulnessScore", scoreText(result.evaluation?.faithfulness));
  setText("#authorityScore", scoreText(result.evaluation?.authority));
  setText("#toolScore", scoreText(result.evaluation?.completeness));
  setText("#latencyScore", "-");
  node.appendChild(el("div", { class: "item eval-card" }, [
    el("strong", { text: "查询改写" }),
    el("div", { class: "tag-row" }, (result.query_rewrites || []).map((query) => el("span", { class: "badge", text: query }))),
  ]));
  node.appendChild(el("div", { class: "item eval-card" }, [
    el("strong", { text: "推理轨迹" }),
    el("div", { class: "list dense mt-12" }, (result.reasoning_trace || []).map((step) => el("div", { class: "item compact" }, [
      el("strong", { text: `${step.stage} · ${step.action}` }),
      el("p", { class: "muted", text: step.output }),
    ]))),
  ]));
  node.appendChild(el("div", { class: "item eval-card" }, [
    el("strong", { text: "答案与自评" }),
    renderScorePills(result.evaluation || {}),
    markdownBlock(result.answer || ""),
  ]));
}

function renderEvaluationPlan(plan) {
  const node = qs("#evaluationPlan");
  clearNode(node);
  (plan.metrics || []).forEach((item) => {
    node.appendChild(el("div", { class: "item compact" }, [
      el("strong", { text: `${item.name} · ${item.metric}` }),
      el("p", { class: "muted", text: item.rule }),
    ]));
  });
  if (plan.release_gate) {
    node.appendChild(el("div", { class: "item compact" }, [
      el("strong", { text: "发布准入" }),
      el("p", { class: "muted", text: Object.entries(plan.release_gate).map(([key, value]) => `${key}: ${value}`).join(" · ") }),
    ]));
  }
}

async function runAgentEval(cases) {
  setBusy("Agent 评测运行中...");
  const data = await apiFetch("/api/training/evaluate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model_path: "current-agent", test_cases: cases }),
  });
  renderEvalResults(data.results);
  await loadEvaluationRuns();
}

async function runRagEval(cases) {
  setBusy("RAG 评测运行中...");
  const data = await apiFetch("/api/evaluation/rag", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ cases }),
  });
  renderRagResults(data.results);
  await loadEvaluationRuns();
}

async function runResearchEval() {
  const testCase = currentCase();
  setBusy("Deep Research 评测运行中...");
  const data = await apiFetch("/api/research/answer", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question: testCase.question, context: { standard_type: qs("#caseCategory").value }, persist_evaluation: true }),
  });
  renderResearch(data.result);
  await loadEvaluationRuns();
}

async function loadEvaluationPlan() {
  try {
    const data = await apiFetch("/api/research/evaluation-plan");
    renderEvaluationPlan(data.plan || {});
  } catch (error) {
    clearNode(qs("#evaluationPlan"));
    qs("#evaluationPlan").appendChild(el("p", { class: "muted", text: `评测指标加载失败：${error.message}` }));
  }
}

async function loadEvaluationRuns() {
  try {
    const data = await apiFetch("/api/evaluation/runs?limit=12");
    renderEvaluationRuns(data.runs || []);
  } catch (error) {
    const node = qs("#evaluationRuns");
    clearNode(node);
    node.appendChild(el("p", { class: "muted", text: `评测历史加载失败：${error.message}` }));
  }
}

async function openEvaluationDeepLink() {
  const runId = new URLSearchParams(window.location.search).get("run_id");
  if (!runId) return;
  try {
    const data = await apiFetch(`/api/evaluation/runs/${encodeURIComponent(runId)}`);
    const run = data.run || {};
    if (run.run_type === "rag") renderRagResults(run.results || {});
    else if (run.run_type === "research") renderResearch(run.results || {});
    else renderEvalResults(run.results || {});
    qs("#evalResults")?.prepend(el("div", { class: "item compact selected-task" }, [
      el("strong", { text: `已打开评测记录 ${runId}` }),
      el("p", { class: "muted", text: `发布门禁：${run.release_gate?.label || run.release_gate?.status || "待复核"}` }),
    ]));
    qs("#training-results")?.scrollIntoView({ behavior: "smooth", block: "start" });
    showToast(`已打开评测记录 ${runId}`, "success");
  } catch (error) {
    showToast(`评测记录打开失败：${error.message}`, "error");
  }
}

function setMode(mode) {
  activeMode = mode;
  qsa("#modeTabs .seg").forEach((button) => button.classList.toggle("active", button.dataset.mode === mode));
}

function bindTrainingPage() {
  renderMetricToggles();
  renderCustomCases();
  loadEvaluationPlan();
  loadEvaluationRuns().then(openEvaluationDeepLink);

  qsa("#modeTabs .seg").forEach((button) => button.addEventListener("click", () => setMode(button.dataset.mode)));

  qs("#addGoldenCase").addEventListener("click", () => {
    const item = currentCase();
    if (!item.question) {
      setBusy("请先填写评测问题。");
      return;
    }
    customCases.push(item);
    renderCustomCases();
  });

  qs("#clearCases").addEventListener("click", () => {
    customCases = [];
    renderCustomCases();
  });

  qs("#runEval").addEventListener("click", async () => {
    try {
      await runAgentEval(customCases.length ? customCases : undefined);
    } catch (error) {
      setBusy(`Agent 评测失败：${error.message}`);
    }
  });

  qs("#runCustomEval").addEventListener("click", async () => {
    try {
      const item = currentCase();
      if (!item.question) {
        setBusy("请先填写评测问题。");
        return;
      }
      if (activeMode === "rag") await runRagEval([item]);
      else if (activeMode === "research") await runResearchEval();
      else await runAgentEval([item]);
    } catch (error) {
      setBusy(`评测失败：${error.message}`);
    }
  });

  qs("#runRagEval").addEventListener("click", async () => {
    try {
      await runRagEval(customCases.length ? customCases : undefined);
    } catch (error) {
      setBusy(`RAG 评测失败：${error.message}`);
    }
  });

  qs("#runResearchEval").addEventListener("click", async () => {
    try {
      await runResearchEval();
    } catch (error) {
      setBusy(`Deep Research 评测失败：${error.message}`);
    }
  });

  qs("#refreshEvalRuns").addEventListener("click", loadEvaluationRuns);
}

document.addEventListener("DOMContentLoaded", bindTrainingPage);

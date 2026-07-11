let selectedTaskId = null;

function priorityBadge(priority) {
  const text = priority || "medium";
  const cls = String(text).toLowerCase() === "high" ? "high" : String(text).toLowerCase() === "low" ? "low" : "medium";
  return el("span", { class: `badge ${cls}`, text });
}

function metricPill(name, value) {
  return el("div", { class: "score-pill" }, [el("span", { text: name }), el("strong", { text: String(value) })]);
}

function compactList(items = [], limit = 3) {
  return items.slice(0, limit).join(" / ");
}

async function loadSkills() {
  const data = await apiFetch("/api/skills");
  const grid = qs("#skillGrid");
  clearNode(grid);
  data.skills.forEach((skill) => {
    grid.appendChild(el("div", { class: "card value-card hover-lift" }, [
      el("strong", { text: skill.title }),
      el("p", { class: "muted", text: skill.description }),
      el("div", { class: "muted", text: `${skill.name} · v${skill.version} · ${(skill.permissions || []).join(", ")}` }),
      el("div", { class: "trace-summary mt-12" }, [
        el("span", { text: `超时 ${skill.resilience?.timeout_seconds || 0}s` }),
        el("span", { text: `缓存 ${skill.resilience?.cache_ttl_seconds || 0}s` }),
        el("span", { text: `熔断 ${skill.resilience?.failure_threshold || 0} 次` })
      ])
    ]));
  });

  const tools = await apiFetch("/api/mcp/tools");
  const mcp = qs("#mcpTools");
  clearNode(mcp);
  const table = el("table");
  table.appendChild(el("thead", {}, [el("tr", {}, ["工具", "说明", "权限", "治理"].map((text) => el("th", { text })))]));
  const body = el("tbody");
  tools.tools.forEach((tool) => body.appendChild(el("tr", {}, [
    el("td", { text: tool.name }),
    el("td", { text: tool.description }),
    el("td", { text: (tool.annotations?.permissions || []).join(", ") }),
    el("td", { text: `${tool.annotations?.timeoutSeconds || 0}s / 缓存 ${tool.annotations?.cacheTtlSeconds || 0}s / 熔断 ${tool.annotations?.failureThreshold || 0}` })
  ])));
  table.appendChild(body);
  mcp.appendChild(table);
}

async function loadObservability() {
  const data = await apiFetch("/api/agent/observability");
  const obs = data.observability || {};
  setText("#taskCount", obs.tasks || 0);
  setText("#toolCalls", obs.tool_calls || 0);
  setText("#toolSuccess", `${Math.round((obs.tool_success_rate || 0) * 100)}%`);
  setText("#avgLatency", `${obs.avg_latency_ms || 0}ms`);
  setText("#blockedTasks", obs.blocked_tasks || 0);
}

async function loadEvolution() {
  const data = await apiFetch("/api/agent/evolution");
  const evolution = data.evolution || {};
  const backlog = evolution.benchmark_backlog || [];
  const proposals = evolution.evolution_proposals || [];
  const control = evolution.evolution_control_plane || {};

  setText("#harnessMaturity", evolution.maturity_score || 0);
  setText("#governanceLanes", control.lanes?.length || 0);
  setText("#backlogCount", backlog.length);
  setText("#proposalCount", proposals.length);

  const backlogNode = qs("#benchmarkBacklog");
  clearNode(backlogNode);
  if (!backlog.length) {
    backlogNode.appendChild(el("div", { class: "empty-compact" }, [el("strong", { text: "暂无新增回归样例" }), el("span", { text: "当前没有高风险回归信号。" })]));
  }
  backlog.forEach((item) => {
    backlogNode.appendChild(el("div", { class: "backlog-item" }, [
      el("div", { class: "item-head" }, [
        el("strong", { text: `${item.case_id} · ${item.source}` }),
        priorityBadge(item.priority)
      ]),
      el("p", { text: item.question }),
      el("small", { class: "muted", text: `验收：${item.acceptance}` })
    ]));
  });

  const proposalNode = qs("#evolutionProposals");
  clearNode(proposalNode);
  proposals.forEach((item) => {
    const button = el("button", { class: "btn", type: "button", text: "物化为任务" });
    button.addEventListener("click", () => materializeProposal(item.proposal_id, button));
    proposalNode.appendChild(el("div", { class: "proposal-card" }, [
      el("div", { class: "item-head" }, [
        el("div", {}, [
          el("strong", { text: `${item.proposal_id} · ${item.title}` }),
          el("small", { class: "muted", text: item.trigger || "" })
        ]),
        priorityBadge(item.priority)
      ]),
      el("p", { text: item.action }),
      el("div", { class: "trace-summary" }, [
        el("span", { text: `验证：${item.validation}` }),
        el("span", { text: `影响：${item.impact}` })
      ]),
      el("div", { class: "toolbar mt-12" }, [button])
    ]));
  });

  const controlNode = qs("#controlPlane");
  clearNode(controlNode);
  controlNode.appendChild(el("div", { class: "control-mode" }, [
    el("strong", { text: `模式：${control.mode || "continuous_improvement"}` }),
    el("span", { class: "badge", text: `可执行提案 ${control.actionable_proposals?.length || 0}` })
  ]));
  (control.lanes || []).forEach((lane, index) => {
    controlNode.appendChild(el("div", { class: "lane-card" }, [
      el("span", { class: "lane-index", text: String(index + 1).padStart(2, "0") }),
      el("div", {}, [
        el("strong", { text: lane.lane }),
        el("small", { class: "muted", text: `${lane.owner} · ${lane.input}` })
      ])
    ]));
  });
}

async function materializeProposal(proposalId, button) {
  button.disabled = true;
  button.textContent = "生成中...";
  try {
    const data = await apiFetch(`/api/agent/evolution/proposals/${proposalId}/task`, { method: "POST" });
    renderTask(data.task);
    await Promise.all([loadTasks(), loadObservability(), loadRuns(), loadEvolution()]);
    showToast(`已生成运行时任务 ${data.task.task_id}`, "success");
  } catch (error) {
    showToast(`提案物化失败：${error.message}`, "error");
  } finally {
    button.disabled = false;
    button.textContent = "物化为任务";
  }
}

async function loadTasks() {
  const data = await apiFetch("/api/agent/tasks?limit=20");
  const node = qs("#taskList");
  clearNode(node);
  if (!data.tasks.length) {
    node.appendChild(el("div", { class: "item muted", text: "暂无运行时任务。" }));
    return;
  }
  data.tasks.forEach((task) => node.appendChild(el("div", { class: "item clickable hover-lift", onclick: () => renderTask(task) }, [
    el("div", { class: "item-head" }, [
      el("strong", { text: `${task.task_id} · ${task.status}` }),
      el("button", {
        class: "btn danger ghost",
        type: "button",
        "data-task-delete": task.task_id,
        text: "删除",
        onclick: (event) => {
          event.stopPropagation();
          deleteTask(task.task_id);
        }
      })
    ]),
    el("div", { class: "muted", text: task.objective }),
    el("div", { class: "trace-summary" }, [
      el("span", { text: `步骤 ${task.steps?.length || 0}/${task.plan?.length || 0}` }),
      el("span", { text: `安全 ${task.safety_gate?.status || "-"}` }),
      el("span", { text: `角色 ${new Set((task.plan || []).map((step) => step.agent_role).filter(Boolean)).size}` })
    ])
  ])));
}

function renderTask(task) {
  selectedTaskId = task.task_id;
  const node = qs("#taskDetail");
  clearNode(node);
  node.appendChild(el("div", { class: "item selected-task" }, [
    el("div", { class: "item-head" }, [
      el("strong", { text: `${task.task_id} · ${task.status}` }),
      el("div", { class: "toolbar" }, [
        el("span", { class: "status-chip", text: task.protocol || "audit-agent-task-v1" }),
        el("button", { class: "btn danger ghost", type: "button", "data-task-delete-detail": task.task_id, text: "删除记录", onclick: () => deleteTask(task.task_id) })
      ])
    ]),
    el("div", { class: "muted", text: task.objective }),
    el("div", { class: "score-grid" }, [
      metricPill("工具调用", task.metrics?.tool_calls || 0),
      metricPill("成功", task.metrics?.successful_tool_calls || 0),
      metricPill("失败", task.metrics?.failed_tool_calls || 0),
      metricPill("平均耗时", `${task.metrics?.avg_latency_ms || 0}ms`)
    ])
  ]));
  (task.plan || []).forEach((step) => {
    const done = (task.steps || []).find((item) => item.step_id === step.step_id);
    node.appendChild(el("details", { class: "item compact eval-detail" }, [
      el("summary", {}, [
        el("strong", { text: `${step.step_id} · ${step.name}` }),
        el("span", { class: "badge", text: done?.status || "待执行" })
      ]),
      el("div", { class: "detail-body" }, [
        el("div", { class: "muted", text: `${step.agent_role || "audit_agent"} · ${step.skill} · ${step.purpose || ""}` }),
        el("small", { class: "muted", text: `依赖：${compactList(step.depends_on || [], 6) || "无"}` })
      ])
    ]));
  });
  (task.artifacts || []).forEach((artifact) => node.appendChild(el("div", { class: "item compact" }, [
    el("strong", { text: `产物 · ${artifact.name}` }),
    el("div", { class: "muted", text: artifact.summary })
  ])));
  (task.reflections || []).forEach((reflection) => node.appendChild(el("div", { class: "item compact" }, [
    el("strong", { text: `反思 · ${reflection.agent_role || reflection.step_id} · ${reflection.verdict}` }),
    el("div", { class: "muted", text: `置信度 ${reflection.confidence} · ${reflection.next_action}` })
  ])));
}

async function deleteTask(taskId) {
  if (!window.confirm(`确认删除任务记录 ${taskId}？此操作会移除本地运行时记录。`)) return;
  await apiFetch(`/api/agent/tasks/${taskId}`, { method: "DELETE" });
  if (selectedTaskId === taskId) {
    selectedTaskId = null;
    const detail = qs("#taskDetail");
    clearNode(detail);
    detail.appendChild(el("p", { class: "muted", text: "记录已删除，请从右侧选择其他任务。" }));
  }
  await Promise.all([loadTasks(), loadObservability(), loadEvolution()]);
  showToast(`已删除任务记录 ${taskId}`, "success");
}

async function createTask() {
  const topics = qs("#taskTopics").value.split(",").map((item) => item.trim()).filter(Boolean);
  const payload = {
    objective: qs("#taskObjective").value,
    context: {
      audit_item: qs("#taskAuditItem").value,
      standard: qs("#taskStandard").value,
      risk_level: qs("#taskRisk").value,
      risk_topics: topics
    }
  };
  const data = await apiFetch("/api/agent/tasks", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  renderTask(data.task);
  await Promise.all([loadTasks(), loadObservability(), loadRuns(), loadEvolution()]);
  showToast(`任务已创建：${data.task.task_id}`, "success");
}

async function runNextStep() {
  if (!selectedTaskId) {
    showToast("请先在右侧选择一个任务。", "error");
    return;
  }
  const data = await apiFetch(`/api/agent/tasks/${selectedTaskId}/run-next`, { method: "POST" });
  renderTask(data.task);
  await Promise.all([loadTasks(), loadObservability(), loadRuns(), loadEvolution()]);
}

async function runSafetyCheck() {
  const data = await apiFetch("/api/safety/check", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ stage: "audit", payload: { text: qs("#safetyPayload").value } })
  });
  const node = qs("#safetyResult");
  clearNode(node);
  const gate = data.gate || {};
  node.appendChild(el("div", { class: "item" }, [
    el("div", { class: "item-head" }, [
      el("strong", { text: `状态：${gate.status}` }),
      el("span", { class: "badge", text: `分数 ${gate.score}` })
    ]),
    el("div", { class: "muted", text: (gate.findings || []).map((item) => item.message).join("；") || "未发现阻断项" })
  ]));
}

async function loadRuns() {
  const node = qs("#skillRuns");
  clearNode(node);
  const data = await apiFetch("/api/skills/runs?limit=12");
  if (!data.runs.length) {
    node.appendChild(el("div", { class: "item muted", text: "暂无 Skill 运行日志。" }));
    return;
  }
  data.runs.forEach((run) => node.appendChild(el("div", { class: "item compact" }, [
    el("div", { class: "item-head" }, [
      el("strong", { text: `${run.run_id} · ${run.skill} · ${run.status}` }),
      el("button", { class: "btn danger ghost", type: "button", "data-run-delete": run.run_id, text: "删除", onclick: () => deleteSkillRun(run.run_id) })
    ]),
    el("div", { class: "muted", text: `${run.duration_ms || 0}ms · ${run.cache_hit ? "缓存命中" : `熔断 ${run.circuit_state || "closed"}`} · ${run.started_at}` })
  ])));
}

async function deleteSkillRun(runId) {
  if (!window.confirm(`确认删除运行日志 ${runId}？`)) return;
  await apiFetch(`/api/skills/runs/${runId}`, { method: "DELETE" });
  await Promise.all([loadRuns(), loadObservability(), loadEvolution()]);
  showToast(`已删除运行日志 ${runId}`, "success");
}

async function bootSkillsPage() {
  await Promise.all([loadSkills(), loadTasks(), loadObservability(), loadRuns(), loadEvolution()]);
}

async function openRuntimeDeepLink() {
  const taskId = new URLSearchParams(window.location.search).get("task_id");
  if (!taskId) return;
  try {
    const data = await apiFetch(`/api/agent/tasks/${encodeURIComponent(taskId)}`);
    renderTask(data.task);
    qs("#taskDetail")?.scrollIntoView({ behavior: "smooth", block: "start" });
    showToast(`已打开运行时任务 ${taskId}`, "success");
  } catch (error) {
    showToast(`运行时任务打开失败：${error.message}`, "error");
  }
}

document.addEventListener("DOMContentLoaded", () => {
  qs("#createTask")?.addEventListener("click", createTask);
  qs("#runNextStep")?.addEventListener("click", runNextStep);
  qs("#runSafetyCheck")?.addEventListener("click", runSafetyCheck);
  qs("#refreshTasks")?.addEventListener("click", () => Promise.all([loadTasks(), loadObservability()]));
  qs("#refreshRuns")?.addEventListener("click", loadRuns);
  qs("#refreshEvolution")?.addEventListener("click", loadEvolution);
  bootSkillsPage()
    .then(openRuntimeDeepLink)
    .catch((error) => showToast(`运行时页面加载失败：${error.message}`, "error"));
});

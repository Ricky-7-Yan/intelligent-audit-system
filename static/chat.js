const CHAT_AGENT_LABELS = {
  planning_agent: "规划 Agent",
  research_agent: "Research Agent",
  evidence_agent: "证据 Agent",
  control_agent: "控制 Agent",
  risk_agent: "风险 Agent",
  compliance_agent: "合规 Agent",
  remediation_agent: "整改 Agent",
  tool_agent: "工具 Agent",
  verifier_agent: "复核 Agent",
  evaluator_agent: "评测 Agent",
  evolution_agent: "自进化 Agent",
  memory_agent: "记忆 Agent",
  audit_agent: "总控 Agent",
};

const CHAT_CONTEXT_LABELS = {
  audit_item: "审计对象",
  standards: "标准",
  audit_type: "审计类型",
  risk_focus: "风险关注",
  scope: "范围",
  systems: "系统",
  period: "期间",
};

const THINKING_STEPS = [
  "识别审计意图与实体",
  "读取会话记忆与相似上下文",
  "调度多角色协作 Agent",
  "检索证据、标准和控制库",
  "执行风险评估与质量门检查",
  "整理结构化结论与可交付内容",
];

let chatSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
let chatMessageCount = 0;

function agentLabel(agent) {
  return CHAT_AGENT_LABELS[agent] || agent || "审计 Agent";
}

function formatValue(value) {
  if (Array.isArray(value)) return value.length ? value.join("、") : "-";
  if (value && typeof value === "object") return JSON.stringify(value, null, 2);
  return value === undefined || value === null || value === "" ? "-" : String(value);
}

function addChatMessage(content, type = "ai", options = {}) {
  const box = qs("#chatBox");
  const message = el("div", { class: `message ${type}${options.pending ? " typing-message" : ""}` });
  if (type === "ai") {
    setMarkdown(message, content);
    if (!options.pending && content) {
      const action = el("button", { class: "message-copy", type: "button", text: "复制" });
      action.addEventListener("click", async () => {
        await navigator.clipboard?.writeText(String(content));
        showToast("回答已复制", "success");
      });
      message.appendChild(action);
    }
  } else {
    message.textContent = content;
  }
  box.appendChild(message);
  box.scrollTop = box.scrollHeight;
  return message;
}

function addThinkingMessage() {
  const box = qs("#chatBox");
  const message = el("div", { class: "message ai typing-message" }, [
    el("div", { class: "typing-title", text: "AuditPilot 正在协同分析" }),
    el("div", { class: "typing-dots" }, [el("span"), el("span"), el("span")]),
    el("div", { class: "typing-step", text: THINKING_STEPS[0] }),
    el("div", { class: "agent-chain mt-12" }, [
      el("span", { class: "agent-chip", text: "规划" }),
      el("span", { class: "agent-chip", text: "证据" }),
      el("span", { class: "agent-chip", text: "控制" }),
      el("span", { class: "agent-chip", text: "风险" }),
      el("span", { class: "agent-chip", text: "复核" }),
    ]),
  ]);
  box.appendChild(message);
  box.scrollTop = box.scrollHeight;
  let index = 0;
  const timer = setInterval(() => {
    if (!document.body.contains(message)) {
      clearInterval(timer);
      return;
    }
    index = (index + 1) % THINKING_STEPS.length;
    setText(".typing-step", THINKING_STEPS[index], message);
  }, 640);
  message._stop = () => clearInterval(timer);
  return message;
}

function removeThinking(message) {
  if (!message) return;
  if (typeof message._stop === "function") message._stop();
  message.remove();
}

function renderContext(context) {
  const node = qs("#auditContext");
  clearNode(node);
  Object.entries(context || {}).forEach(([key, value]) => {
    node.appendChild(el("div", { class: "item compact" }, [
      el("strong", { text: CHAT_CONTEXT_LABELS[key] || key }),
      el("div", { class: "muted", text: formatValue(value) }),
    ]));
  });
  if (!node.childElementCount) node.appendChild(el("p", { class: "muted", text: "暂无上下文" }));
}

function renderRisk(result) {
  const node = qs("#riskPanel");
  clearNode(node);
  const risk = result.risk_assessment;
  const quality = result.quality_gate;
  if (!risk) {
    node.appendChild(el("p", { class: "muted", text: "暂无风险评估" }));
    return;
  }
  node.appendChild(el("div", { class: "item" }, [
    el("div", { class: "item-head" }, [
      el("strong", { text: "剩余风险" }),
      riskBadge(risk.risk_level),
    ]),
    el("div", { class: "score-grid mt-12" }, [
      el("div", { class: "score-pill" }, [el("span", { text: "风险评分" }), el("strong", { text: formatValue(risk.risk_score) })]),
      el("div", { class: "score-pill" }, [el("span", { text: "控制抵减" }), el("strong", { text: formatValue(risk.control_reduction || 0) })]),
    ]),
  ]));
  if (quality) {
    node.appendChild(el("div", { class: "item" }, [
      el("div", { class: "item-head" }, [
        el("strong", { text: "质量门" }),
        riskBadge(quality.status === "passed" ? "通过" : "需复核"),
      ]),
      el("div", { class: "muted", text: `置信度：${formatValue(quality.confidence)}；缺失证据：${formatValue(quality.missing_evidence)}` }),
    ]));
  }
}

function renderRoutingAndMemory(data) {
  const node = qs("#routingPanel");
  clearNode(node);
  const routing = data.routing || {};
  const agents = routing.agents && routing.agents.length ? routing.agents : ["audit_agent"];
  const plan = routing.collaboration_plan || [];
  node.appendChild(el("div", { class: "item route-summary" }, [
    el("div", { class: "item-head" }, [
      el("strong", { text: `${routing.intent || "general_audit"} · ${Math.round((routing.confidence || 0) * 100)}%` }),
      riskBadge(routing.urgency === "critical" ? "紧急" : routing.multi_agent ? "多 Agent" : "单 Agent"),
    ]),
    el("div", { class: "agent-chain mt-12" }, agents.map((agent, index) => el("span", { class: "agent-chip", text: `${index + 1}. ${agentLabel(agent)}` }))),
    el("div", { class: "tag-row mt-12" }, [
      el("span", { class: "badge", text: `${agents.length} 个协作角色` }),
      el("span", { class: "badge", text: routing.strategy || "hybrid-routing" }),
    ]),
  ]));
  if (plan.length) {
    node.appendChild(el("div", { class: "route-plan" }, plan.map((step, index) => el("div", { class: "route-step" }, [
      el("span", { class: "route-index", text: String(index + 1).padStart(2, "0") }),
      el("div", {}, [
        el("strong", { text: agentLabel(step.agent) }),
        el("p", { class: "muted", text: step.responsibility || "补充审计判断与质量复核。" }),
      ]),
    ]))));
  }
  const layers = data.memory?.memory_layers || data.persistent_memory?.memory_layers || {};
  setText("#workingMemory", layers.working ?? data.memory?.working_messages ?? 0);
  setText("#episodicMemory", layers.episodic ?? data.memory?.episodes ?? 0);
  setText("#profileFields", layers.profile_fields ?? Object.keys(data.memory?.profile || {}).length);
}

async function sendChatMessage() {
  const input = qs("#messageInput");
  const message = input.value.trim();
  if (!message) return;
  addChatMessage(message, "user");
  input.value = "";
  input.disabled = true;
  qs("#sendBtn").disabled = true;
  const thinking = addThinkingMessage();
  const started = performance.now();
  try {
    const data = await apiFetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json; charset=utf-8" },
      body: JSON.stringify({ message, session_id: chatSessionId, llm_enhance: false }),
    });
    removeThinking(thinking);
    addChatMessage(data.response || "已完成分析，但未返回文本回答。", "ai");
    renderContext(data.audit_context);
    renderRisk(data);
    renderRoutingAndMemory(data);
    chatMessageCount += 2;
    setText("#messageCount", chatMessageCount);
    showToast(`协作分析完成 · ${Math.round(performance.now() - started)}ms`, "success");
  } catch (error) {
    removeThinking(thinking);
    addChatMessage(`请求失败：${error.message}`, "ai");
    showToast(`请求失败：${error.message}`, "error");
  } finally {
    input.disabled = false;
    qs("#sendBtn").disabled = false;
    input.focus();
  }
}

document.addEventListener("DOMContentLoaded", () => {
  setText("#sessionId", chatSessionId);
  qs("#sendBtn")?.addEventListener("click", sendChatMessage);
  qs("#messageInput")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter") sendChatMessage();
  });
  qs("#clearBtn")?.addEventListener("click", () => {
    chatSessionId = `session_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    chatMessageCount = 0;
    setText("#sessionId", chatSessionId);
    setText("#messageCount", 0);
    clearNode(qs("#chatBox"));
    addChatMessage("新会话已创建。你可以继续输入审计对象、标准或风险场景。", "ai");
    renderContext({});
    renderRisk({});
    clearNode(qs("#routingPanel"));
    qs("#routingPanel").appendChild(el("p", { class: "muted", text: "发送消息后显示意图、置信度、协作 Agent 和每个角色的分工。" }));
  });
  qsa("[data-quick]").forEach((button) => button.addEventListener("click", () => {
    qs("#messageInput").value = button.dataset.quick;
    sendChatMessage();
  }));
});

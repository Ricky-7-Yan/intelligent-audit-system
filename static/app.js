function qs(selector, root = document) {
  return root.querySelector(selector);
}

function qsa(selector, root = document) {
  return Array.from(root.querySelectorAll(selector));
}

function setText(selector, value, root = document) {
  const node = qs(selector, root);
  if (node) node.textContent = value;
}

function clearNode(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  });
  children.forEach((child) => node.appendChild(typeof child === "string" ? document.createTextNode(child) : child));
  return node;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json().catch(() => ({}));
  if (!response.ok || data.success === false) {
    throw new Error(data.detail || data.error || `请求失败: ${response.status}`);
  }
  return data;
}

function riskBadge(level) {
  const normalized = String(level || "").toLowerCase();
  const cls = level === "高" || normalized === "high" ? "high" : level === "中" || normalized === "medium" ? "medium" : "low";
  return el("span", { class: `badge ${cls}`, text: level || "低" });
}

async function loadHealth() {
  try {
    const data = await apiFetch("/api/health");
    const services = data.services || {};
    setText("#healthText", `RAG ${services.rag_documents || 0} 条知识 · LLM ${services.llm ? "在线" : "降级"}`);
  } catch {
    setText("#healthText", "服务状态待确认");
  }
}

document.addEventListener("DOMContentLoaded", loadHealth);

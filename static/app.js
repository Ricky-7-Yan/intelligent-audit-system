const API_BASE = window.location.protocol === "file:" ? "http://127.0.0.1:8000" : "";

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
  if (!node) return;
  while (node.firstChild) node.removeChild(node.firstChild);
}

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value);
  });
  children.forEach((child) => node.appendChild(typeof child === "string" ? document.createTextNode(child) : child));
  return node;
}

function apiUrl(url) {
  if (url.startsWith("http")) return url;
  return `${API_BASE}${url}`;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(apiUrl(url), options);
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

function serviceUrl(path) {
  return `${API_BASE}${path}`;
}

async function loadHealth() {
  if (window.location.protocol === "file:") {
    setText("#healthText", "请通过 http://127.0.0.1:8000 访问");
  }
  try {
    const data = await apiFetch("/api/health");
    const services = data.services || {};
    setText("#healthText", `RAG ${services.rag_documents || 0} 条知识 · LLM ${services.llm ? "在线" : "降级"}`);
  } catch {
    setText("#healthText", window.location.protocol === "file:" ? "本地文件模式：请先启动服务" : "服务状态待确认");
  }
}

document.addEventListener("DOMContentLoaded", loadHealth);

document.addEventListener("DOMContentLoaded", () => {
  if (window.location.protocol !== "file:") return;
  qsa("a[href^='/']").forEach((anchor) => {
    anchor.href = `${API_BASE}${anchor.getAttribute("href")}`;
  });
});

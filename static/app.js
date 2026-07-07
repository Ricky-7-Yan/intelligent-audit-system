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
    else if (key === "style") node.setAttribute("style", value);
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (value === true) node.setAttribute(key, key);
    else if (value !== false && value !== null && value !== undefined) node.setAttribute(key, value);
  });
  children.forEach((child) => node.appendChild(typeof child === "string" ? document.createTextNode(child) : child));
  return node;
}

function escapeHtml(value = "") {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function renderInlineMarkdown(value = "") {
  let text = escapeHtml(value);
  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return text;
}

function isMarkdownTableSeparator(line = "") {
  return /^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(line);
}

function splitMarkdownRow(line = "") {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function markdownToHtml(markdown = "") {
  const lines = String(markdown || "")
    .replace(/\r\n/g, "\n")
    .split("\n");
  const html = [];
  let paragraph = [];
  let list = null;

  function flushParagraph() {
    if (!paragraph.length) return;
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "))}</p>`);
    paragraph = [];
  }

  function closeList() {
    if (!list) return;
    html.push(`</${list}>`);
    list = null;
  }

  for (let index = 0; index < lines.length; index += 1) {
    const raw = lines[index];
    const line = raw.trim();
    const next = lines[index + 1] || "";

    if (!line || /^-{3,}$/.test(line)) {
      flushParagraph();
      closeList();
      continue;
    }

    if (line.includes("|") && isMarkdownTableSeparator(next)) {
      flushParagraph();
      closeList();
      const headers = splitMarkdownRow(line);
      index += 1;
      const bodyRows = [];
      while (index + 1 < lines.length && lines[index + 1].trim().includes("|")) {
        index += 1;
        bodyRows.push(splitMarkdownRow(lines[index]));
      }
      html.push(
        `<div class="markdown-table-wrap"><table class="markdown-table"><thead><tr>${headers
          .map((cell) => `<th>${renderInlineMarkdown(cell)}</th>`)
          .join("")}</tr></thead><tbody>${bodyRows
          .map((row) => `<tr>${headers.map((_, cellIndex) => `<td>${renderInlineMarkdown(row[cellIndex] || "")}</td>`).join("")}</tr>`)
          .join("")}</tbody></table></div>`
      );
      continue;
    }

    const heading = /^(#{1,4})\s+(.+)$/.exec(line);
    if (heading) {
      flushParagraph();
      closeList();
      const level = Math.min(heading[1].length + 1, 5);
      html.push(`<h${level}>${renderInlineMarkdown(heading[2])}</h${level}>`);
      continue;
    }

    const unordered = /^[-*]\s+(.+)$/.exec(line);
    if (unordered) {
      flushParagraph();
      if (list !== "ul") {
        closeList();
        list = "ul";
        html.push("<ul>");
      }
      html.push(`<li>${renderInlineMarkdown(unordered[1])}</li>`);
      continue;
    }

    const ordered = /^\d+[.)]\s+(.+)$/.exec(line);
    if (ordered) {
      flushParagraph();
      if (list !== "ol") {
        closeList();
        list = "ol";
        html.push("<ol>");
      }
      html.push(`<li>${renderInlineMarkdown(ordered[1])}</li>`);
      continue;
    }

    closeList();
    paragraph.push(line);
  }
  flushParagraph();
  closeList();
  return html.join("");
}

function setMarkdown(target, markdown) {
  const node = typeof target === "string" ? qs(target) : target;
  if (!node) return;
  node.classList.add("markdown-body");
  node.innerHTML = markdownToHtml(markdown);
}

function showToast(message, tone = "info") {
  let host = qs("#toastHost");
  if (!host) {
    host = el("div", { id: "toastHost", class: "toast-host" });
    document.body.appendChild(host);
  }
  const item = el("div", { class: `toast ${tone}`, text: message });
  host.appendChild(item);
  setTimeout(() => item.classList.add("show"), 20);
  setTimeout(() => {
    item.classList.remove("show");
    setTimeout(() => item.remove(), 220);
  }, 2800);
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
    setText("#healthText", `RAG ${services.rag_documents || 0} 条知识 · LLM ${services.llm ? "已配置" : "降级"}`);
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

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

function typeLabel(type) {
  return {
    command: "入口",
    audit: "审计",
    evaluation: "评测",
    evidence: "证据",
    task: "任务",
    tool: "工具",
  }[type] || "结果";
}

function initCommandCenter() {
  if (qs("#commandCenter")) return;
  let activeIndex = 0;
  let latestResults = [];
  let searchTimer = null;

  const button = el("button", { id: "commandLauncher", class: "command-launcher", type: "button" }, [
    el("span", { text: "⌘K" }),
    el("strong", { text: "全局搜索" }),
  ]);
  const overlay = el("div", { id: "commandCenter", class: "command-overlay hidden", role: "dialog", "aria-modal": "true" }, [
    el("div", { class: "command-panel-wrap" }, [
      el("div", { class: "command-input-row" }, [
        el("span", { class: "command-icon", text: "⌕" }),
        el("input", { id: "commandInput", type: "search", placeholder: "搜索审计项目、评测记录、证据、运行任务或页面入口…" }),
        el("button", { class: "btn ghost", type: "button", id: "commandClose", text: "Esc" }),
      ]),
      el("div", { class: "command-hint", text: "支持 Ctrl/⌘ + K 唤起；回车打开选中结果。" }),
      el("div", { id: "commandResults", class: "command-results" }),
    ]),
  ]);
  document.body.appendChild(button);
  document.body.appendChild(overlay);

  const input = qs("#commandInput");
  const resultsNode = qs("#commandResults");

  function renderResults(results = [], query = "") {
    latestResults = results;
    activeIndex = Math.min(activeIndex, Math.max(results.length - 1, 0));
    clearNode(resultsNode);
    if (!results.length) {
      resultsNode.appendChild(el("div", { class: "command-empty" }, [
        el("strong", { text: query ? "没有匹配结果" : "输入关键词开始检索" }),
        el("span", { text: query ? "可以换一个审计对象、运行编号、证据文件或功能关键词试试。" : "例如：ERP、权限、评测、Harness、证据、RAG。" }),
      ]));
      return;
    }
    results.forEach((item, index) => {
      const row = el("button", { class: `command-result ${index === activeIndex ? "active" : ""}`, type: "button" }, [
        el("span", { class: `command-type ${item.type || "result"}`, text: typeLabel(item.type) }),
        el("span", { class: "command-copy" }, [
          el("strong", { text: item.title || "未命名结果" }),
          el("small", { text: item.subtitle || item.href || "" }),
        ]),
        el("span", { class: "command-badge", text: item.badge || "打开" }),
      ]);
      row.addEventListener("mousemove", () => {
        activeIndex = index;
        renderResults(latestResults, input.value);
      });
      row.addEventListener("click", () => openResult(item));
      resultsNode.appendChild(row);
    });
  }

  async function runSearch(query = "") {
    try {
      const data = await apiFetch(`/api/search?q=${encodeURIComponent(query)}&limit=12`);
      renderResults(data.results || [], query);
    } catch (error) {
      clearNode(resultsNode);
      resultsNode.appendChild(el("div", { class: "command-empty" }, [
        el("strong", { text: "搜索暂不可用" }),
        el("span", { text: error.message }),
      ]));
    }
  }

  function openCenter() {
    overlay.classList.remove("hidden");
    document.body.classList.add("command-open");
    input.focus();
    input.select();
    runSearch(input.value.trim());
  }

  function closeCenter() {
    overlay.classList.add("hidden");
    document.body.classList.remove("command-open");
  }

  function openResult(item) {
    if (!item?.href) return;
    closeCenter();
    window.location.href = item.href.startsWith("/") ? serviceUrl(item.href) : item.href;
  }

  button.addEventListener("click", openCenter);
  qs("#commandClose").addEventListener("click", closeCenter);
  overlay.addEventListener("click", (event) => {
    if (event.target === overlay) closeCenter();
  });
  input.addEventListener("input", () => {
    clearTimeout(searchTimer);
    const query = input.value.trim();
    searchTimer = setTimeout(() => runSearch(query), 120);
  });
  document.addEventListener("keydown", (event) => {
    const target = event.target;
    const isTyping = target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName);
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      openCenter();
      return;
    }
    if (overlay.classList.contains("hidden")) return;
    if (event.key === "Escape") {
      event.preventDefault();
      closeCenter();
    } else if (event.key === "ArrowDown") {
      event.preventDefault();
      activeIndex = Math.min(activeIndex + 1, latestResults.length - 1);
      renderResults(latestResults, input.value);
    } else if (event.key === "ArrowUp") {
      event.preventDefault();
      activeIndex = Math.max(activeIndex - 1, 0);
      renderResults(latestResults, input.value);
    } else if (event.key === "Enter" && latestResults[activeIndex]) {
      event.preventDefault();
      openResult(latestResults[activeIndex]);
    } else if (!isTyping) {
      input.focus();
    }
  });
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
  const cls = ["高", "紧急", "需复核", "blocked", "critical", "high"].includes(level) || ["critical", "high", "blocked"].includes(normalized)
    ? "high"
    : ["中", "多 Agent", "warning", "medium", "needs_review"].includes(level) || ["medium", "warning", "needs_review"].includes(normalized)
      ? "medium"
      : "low";
  return el("span", { class: `badge ${cls}`, text: level || "低" });
}

function serviceUrl(path) {
  return `${API_BASE}${path}`;
}

function initSectionNavigator() {
  const main = qs("main.main");
  if (!main || qs("[data-generated-section-nav]")) return;
  const sections = qsa("section[data-section-label]", main);
  if (!sections.length) return;
  let nav = qs("[data-section-nav]", main);
  if (!nav) {
    nav = el("div", { class: "page-quickbar", "data-section-nav": "", "data-generated-section-nav": "true" });
    const topbar = qs(".topbar", main);
    if (topbar && topbar.nextSibling) main.insertBefore(nav, topbar.nextSibling);
    else main.prepend(nav);
  }
  clearNode(nav);
  let focusMode = false;
  let activeSection = sections[0];
  const applyFocus = (section = activeSection) => {
    activeSection = section || activeSection;
    sections.forEach((item) => item.classList.toggle("section-focus-hidden", focusMode && item !== activeSection));
    document.body.classList.toggle("section-focus-mode", focusMode);
  };
  const label = el("span", { class: "quickbar-label", text: "页面导航" });
  nav.appendChild(label);
  function sectionDisplayLabel(section) {
    if (qs("#evidenceAnalysisResult", section)) return "证据";
    if (qs("#auditResult", section)) return "结论";
    const labels = {
      "audit-command": "指挥",
      "audit-kpis": "指标",
      "audit-research-section": "Research",
      "audit-delivery-section": "交付",
      "audit-evidence-section": "证据",
      "audit-plan-section": "计划",
      "audit-workbench-section": "工作台",
      "audit-control-section": "控制",
      "audit-program-section": "程序",
      "audit-findings-section": "发现",
      "audit-remediation-section": "整改",
      "audit-tasks-section": "跟踪",
      "audit-review-section": "复核",
      "training-hero": "总览",
      "training-design": "用例",
      "training-kpis": "指标",
      "training-results": "结果",
      "training-history": "历史",
    };
    const raw = section.dataset.sectionLabel || "";
    if (labels[section.id]) return labels[section.id];
    if (!raw || raw.includes("�") || raw.includes("□")) return section.querySelector(".panel-title, h2")?.textContent?.trim() || "分区";
    return raw;
  }
  sections.forEach((section, index) => {
    if (!section.id) section.id = `section-${index + 1}`;
    section.dataset.sectionLabel = sectionDisplayLabel(section);
    const button = el("button", { class: "quickbar-link", type: "button", text: section.dataset.sectionLabel });
    button.addEventListener("click", () => {
      activeSection = section;
      if (focusMode) applyFocus(section);
      section.scrollIntoView({ behavior: "smooth", block: "start" });
    });
    nav.appendChild(button);
    if (section.dataset.collapsible === "true" && !qs(".section-collapse-btn", section)) {
      if (section.dataset.defaultCollapsed === "true") {
        section.classList.add("section-collapsed");
      }
      const collapse = el("button", {
        class: "section-collapse-btn",
        type: "button",
        text: section.classList.contains("section-collapsed") ? "展开" : "收起",
        "aria-expanded": section.classList.contains("section-collapsed") ? "false" : "true",
      });
      collapse.addEventListener("click", () => {
        section.classList.toggle("section-collapsed");
        collapse.textContent = section.classList.contains("section-collapsed") ? "展开" : "收起";
        collapse.setAttribute("aria-expanded", section.classList.contains("section-collapsed") ? "false" : "true");
      });
      section.appendChild(collapse);
    }
  });
  const density = el("button", { class: "quickbar-link density-toggle", type: "button", text: "紧凑视图" });
  density.addEventListener("click", () => {
    document.body.classList.toggle("density-compact");
    density.textContent = document.body.classList.contains("density-compact") ? "舒展视图" : "紧凑视图";
  });
  nav.appendChild(density);
  const focus = el("button", { class: "quickbar-link focus-toggle", type: "button", text: "专注视图" });
  focus.addEventListener("click", () => {
    focusMode = !focusMode;
    focus.textContent = focusMode ? "显示全部" : "专注视图";
    applyFocus(activeSection);
  });
  nav.appendChild(focus);
  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    activeSection = visible.target;
    qsa(".quickbar-link", nav).forEach((button) => {
      button.classList.toggle("active", button.textContent === visible.target.dataset.sectionLabel);
    });
  }, { rootMargin: "-20% 0px -70% 0px", threshold: [0.05, 0.2, 0.6] });
  sections.forEach((section) => observer.observe(section));
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
document.addEventListener("DOMContentLoaded", initSectionNavigator);
document.addEventListener("DOMContentLoaded", initCommandCenter);

document.addEventListener("DOMContentLoaded", () => {
  if (window.location.protocol !== "file:") return;
  qsa("a[href^='/']").forEach((anchor) => {
    anchor.href = `${API_BASE}${anchor.getAttribute("href")}`;
  });
});

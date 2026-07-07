"""Deterministic hybrid intent routing for enterprise audit workflows.

The router combines domain patterns with a local character n-gram similarity
signal.  It remains available when no external model is configured and exposes
the component scores so routing decisions are auditable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List


@dataclass(frozen=True)
class IntentDefinition:
    name: str
    agent: str
    keywords: tuple[str, ...]
    examples: tuple[str, ...]


INTENTS: tuple[IntentDefinition, ...] = (
    IntentDefinition(
        "scope_planning",
        "planning_agent",
        ("范围", "计划", "审计对象", "审计期间", "重点问题", "立项"),
        ("为系统制定审计范围和执行计划", "如何启动一次审计项目"),
    ),
    IntentDefinition(
        "evidence_analysis",
        "evidence_agent",
        ("证据", "取证", "日志", "样本", "底稿", "材料", "上传"),
        ("分析审计证据并识别缺口", "需要收集哪些日志和审批材料"),
    ),
    IntentDefinition(
        "control_testing",
        "control_agent",
        ("控制", "测试", "抽样", "穿行", "职责分离", "ITGC"),
        ("设计控制测试和抽样程序", "检查职责分离控制是否有效"),
    ),
    IntentDefinition(
        "risk_assessment",
        "risk_agent",
        ("风险", "异常", "高风险", "影响", "可能性", "缺陷"),
        ("评估剩余风险和业务影响", "识别系统中的关键风险"),
    ),
    IntentDefinition(
        "compliance_mapping",
        "compliance_agent",
        ("ISO27001", "SOX", "COBIT", "合规", "标准", "条款", "数据安全法"),
        ("把审计事项映射到合规标准", "依据 ISO27001 检查访问控制"),
    ),
    IntentDefinition(
        "finding_remediation",
        "remediation_agent",
        ("发现", "整改", "责任人", "到期", "关闭", "复核", "建议"),
        ("形成审计发现并推进整改闭环", "如何验证整改是否可以关闭"),
    ),
    IntentDefinition(
        "knowledge_query",
        "research_agent",
        ("什么是", "怎么做", "查询", "知识", "制度", "历史", "参考"),
        ("查询制度知识和历史审计经验", "检索相关审计依据"),
    ),
)


class HybridIntentRouter:
    """Route audit requests with explainable pattern and semantic signals."""

    pattern_weight = 0.62
    semantic_weight = 0.38

    def classify(self, message: str) -> Dict[str, Any]:
        normalized = self._normalize(message)
        scored: List[Dict[str, Any]] = []
        for definition in INTENTS:
            matched = [keyword for keyword in definition.keywords if keyword.lower() in normalized]
            pattern = min(1.0, len(matched) / 2)
            semantic = max(
                (self._cosine(self._vector(normalized), self._vector(example)) for example in definition.examples),
                default=0.0,
            )
            score = self.pattern_weight * pattern + self.semantic_weight * semantic
            scored.append(
                {
                    "intent": definition.name,
                    "agent": definition.agent,
                    "score": round(score, 4),
                    "signals": {"pattern": round(pattern, 4), "semantic": round(semantic, 4)},
                    "matched_keywords": matched,
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        primary = scored[0]
        if primary["score"] < 0.14:
            primary = {
                "intent": "general_audit",
                "agent": "audit_agent",
                "score": 0.35,
                "signals": {"pattern": 0.0, "semantic": 0.0},
                "matched_keywords": [],
            }

        collaboration_floor = max(0.25, primary["score"] * 0.68)
        selected = [
            item for item in scored
            if item["score"] >= collaboration_floor and item["intent"] != primary["intent"]
        ][:4]
        agents = self._complete_collaboration([primary["agent"], *[item["agent"] for item in selected]], message, primary["intent"])
        confidence = min(0.98, 0.42 + primary["score"] * 0.54)
        return {
            "intent": primary["intent"],
            "confidence": round(confidence, 3),
            "urgency": self._urgency(message),
            "agents": agents,
            "multi_agent": len(agents) > 1,
            "collaboration_plan": self._collaboration_plan(agents),
            "entities": self._entities(message),
            "decision": primary,
            "alternatives": scored[1:4],
            "strategy": "pattern+local_ngram_similarity",
        }

    def _complete_collaboration(self, agents: List[str], message: str, primary_intent: str) -> List[str]:
        default_chain = ["planning_agent", "evidence_agent", "control_agent", "risk_agent", "compliance_agent", "remediation_agent"]
        if any(term in message.lower() for term in ("research", "deep", "研究", "查询", "资料", "jd")):
            default_chain.insert(2, "research_agent")
        if any(term in message.lower() for term in ("工具", "tool", "下载", "报告", "导出", "运行")):
            default_chain.append("tool_agent")
        target_size = 5 if primary_intent != "general_audit" else 4
        merged = list(dict.fromkeys([*agents, *default_chain]))
        return merged[: max(target_size, min(len(merged), 6))]

    def _collaboration_plan(self, agents: List[str]) -> List[Dict[str, str]]:
        responsibilities = {
            "planning_agent": "拆解审计目标、范围、任务顺序和交付物。",
            "evidence_agent": "识别证据缺口、样本需求、日志与底稿路径。",
            "research_agent": "执行 Deep Research、查询改写和来源融合。",
            "control_agent": "映射控制库、设计控制测试与抽样方案。",
            "risk_agent": "评估风险等级、影响、概率和剩余风险。",
            "compliance_agent": "对齐 ISO/SOX/COBIT 等标准条款与合规要求。",
            "remediation_agent": "生成整改责任、关闭标准、复核门槛和跟踪计划。",
            "tool_agent": "选择可执行工具、导出报告并维护运行轨迹。",
            "audit_agent": "统筹完整审计结论与结构化输出。",
        }
        return [{"agent": agent, "responsibility": responsibilities.get(agent, "补充审计判断与质量复核。")} for agent in agents]

    def _normalize(self, text: str) -> str:
        return re.sub(r"\s+", "", text.lower())

    def _vector(self, text: str) -> Counter[str]:
        compact = self._normalize(text)
        features: Counter[str] = Counter()
        for token in re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", compact):
            features[f"t:{token}"] += 1
        for size in (2, 3):
            for index in range(max(0, len(compact) - size + 1)):
                features[f"g{size}:{compact[index:index + size]}"] += 1
        return features

    def _cosine(self, left: Counter[str], right: Counter[str]) -> float:
        if not left or not right:
            return 0.0
        shared = set(left) & set(right)
        numerator = sum(left[key] * right[key] for key in shared)
        left_norm = math.sqrt(sum(value * value for value in left.values()))
        right_norm = math.sqrt(sum(value * value for value in right.values()))
        return numerator / max(left_norm * right_norm, 1e-9)

    def _urgency(self, text: str) -> str:
        critical = ("立即", "马上", "重大", "泄露", "舞弊", "生产事故", "监管")
        high = ("紧急", "高风险", "逾期", "投诉", "阻断")
        if any(keyword in text for keyword in critical):
            return "critical"
        if any(keyword in text for keyword in high):
            return "high"
        return "normal"

    def _entities(self, text: str) -> Dict[str, List[str]]:
        standards = [item for item in ("ISO27001", "SOX", "COBIT", "数据安全法") if item.lower() in text.lower()]
        systems = self._unique(
            [
                *re.findall(r"[\w\u4e00-\u9fff-]{1,24}(?:系统|平台|数据库)", text, flags=re.IGNORECASE),
                *re.findall(r"\b(?:ERP|CRM|OA)\b", text, flags=re.IGNORECASE),
            ]
        )
        control_ids = self._unique(re.findall(r"\b[A-Z]{2,8}-\d{1,4}\b", text.upper()))
        return {"standards": standards, "systems": systems[:6], "control_ids": control_ids[:8]}

    def _unique(self, values: Iterable[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

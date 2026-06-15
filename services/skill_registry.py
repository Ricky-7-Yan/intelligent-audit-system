"""Agent skill registry and MCP-style tool facade.

The implementation is intentionally local and dependency-light, but the surface
resembles enterprise Agent platforms: skills expose schemas, permissions,
execution records and MCP-style tool descriptors.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from config import PATHS


@dataclass
class Skill:
    name: str
    title: str
    description: str
    input_schema: Dict[str, Any]
    permissions: List[str]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    version: str = "1.0.0"


class SkillRegistry:
    def __init__(self) -> None:
        self.skills: Dict[str, Skill] = {}
        self.log_file = PATHS["data"] / "skill_runs" / "runs.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._register_builtin_skills()

    def list_skills(self) -> List[Dict[str, Any]]:
        return [self._describe(skill) for skill in self.skills.values()]

    def mcp_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "description": skill.description,
                "inputSchema": skill.input_schema,
                "annotations": {"title": skill.title, "permissions": skill.permissions, "version": skill.version},
            }
            for skill in self.skills.values()
        ]

    def execute(self, name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if name not in self.skills:
            raise KeyError(name)
        skill = self.skills[name]
        run_id = f"SK-{uuid.uuid4().hex[:10].upper()}"
        started = datetime.now().isoformat()
        try:
            result = skill.handler(payload)
            status = "success"
        except Exception as exc:
            result = {"error": str(exc)}
            status = "failed"
        record = {
            "run_id": run_id,
            "skill": name,
            "status": status,
            "input": payload,
            "output": result,
            "started_at": started,
            "finished_at": datetime.now().isoformat(),
        }
        self.log_file.open("a", encoding="utf-8").write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        lines = self.log_file.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()][::-1]

    def _register(self, skill: Skill) -> None:
        self.skills[skill.name] = skill

    def _describe(self, skill: Skill) -> Dict[str, Any]:
        return {
            "name": skill.name,
            "title": skill.title,
            "description": skill.description,
            "input_schema": skill.input_schema,
            "permissions": skill.permissions,
            "version": skill.version,
        }

    def _register_builtin_skills(self) -> None:
        self._register(
            Skill(
                name="audit.scope_planner",
                title="审计范围规划",
                description="根据审计对象、系统、标准和风险主题生成可执行审计范围。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "audit_item": {"type": "string"},
                        "standard": {"type": "string"},
                        "risk_topics": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["audit_item"],
                },
                permissions=["read:standards"],
                handler=self._scope_planner,
            )
        )
        self._register(
            Skill(
                name="audit.evidence_checklist",
                title="证据清单生成",
                description="为审计控制和风险主题生成证据清单、取证方式和缺口提示。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "control_domain": {"type": "string"},
                        "risk_level": {"type": "string"},
                    },
                    "required": ["control_domain"],
                },
                permissions=["read:controls"],
                handler=self._evidence_checklist,
            )
        )
        self._register(
            Skill(
                name="audit.finding_writer",
                title="审计发现草稿",
                description="根据现状、标准、原因和影响生成审计发现五要素草稿。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "condition": {"type": "string"},
                        "criteria": {"type": "string"},
                        "risk": {"type": "string"},
                    },
                    "required": ["condition", "criteria"],
                },
                permissions=["write:workpaper"],
                handler=self._finding_writer,
            )
        )
        self._register(
            Skill(
                name="rag.query",
                title="RAG 知识检索",
                description="查询审计知识库并返回可引用来源。",
                input_schema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
                permissions=["read:knowledge"],
                handler=self._rag_query,
            )
        )
        self._register(
            Skill(
                name="audit.control_mapper",
                title="控制矩阵映射",
                description="根据风险主题、标准和审计对象生成可执行控制测试矩阵。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "audit_item": {"type": "string"},
                        "standard": {"type": "string"},
                        "risk_topics": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["audit_item"],
                },
                permissions=["read:controls", "write:workpaper"],
                handler=self._control_mapper,
            )
        )
        self._register(
            Skill(
                name="agent.eval_case_designer",
                title="Agent 评测用例设计",
                description="面向 Agent / RAG / 工具调用场景生成可落地的评测用例。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "capabilities": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["scenario"],
                },
                permissions=["write:evaluation"],
                handler=self._eval_case_designer,
            )
        )
        self._register(
            Skill(
                name="audit.remediation_planner",
                title="整改任务生成",
                description="把审计发现转化为责任人、到期时间、验收指标和跟踪状态。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "finding": {"type": "string"},
                        "severity": {"type": "string"},
                        "owner_role": {"type": "string"},
                    },
                    "required": ["finding"],
                },
                permissions=["write:tasks"],
                handler=self._remediation_planner,
            )
        )

    def _scope_planner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = payload.get("audit_item", "待审计对象")
        standard = payload.get("standard", "ISO27001/COBIT")
        topics = payload.get("risk_topics") or ["权限", "变更", "日志", "数据"]
        return {
            "scope": f"{item} 的关键流程、权限、变更、日志和数据处理活动",
            "standard": standard,
            "risk_topics": topics,
            "deliverables": ["审计范围说明", "控制矩阵", "抽样计划", "发现草稿", "整改任务"],
        }

    def _evidence_checklist(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        domain = payload.get("control_domain", "访问控制")
        common = ["制度文件", "审批记录", "系统配置截图", "日志样本", "抽样底稿", "复核记录"]
        if "变更" in domain:
            common.extend(["变更单", "测试报告", "回退方案"])
        if "数据" in domain:
            common.extend(["数据目录", "分类分级规则", "加密/脱敏配置"])
        return {"control_domain": domain, "evidence": common, "collection_method": "系统导出 + 访谈 + 抽样核验"}

    def _finding_writer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        condition = payload.get("condition", "")
        criteria = payload.get("criteria", "")
        risk = payload.get("risk", "可能影响控制有效性")
        return {
            "title": "控制执行证据不足",
            "condition": condition,
            "criteria": criteria,
            "cause": "控制责任、系统留痕或复核机制不完整",
            "effect": risk,
            "recommendation": "补齐控制证据并建立周期性复核和例外跟踪机制",
        }

    def _rag_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from rag.agentic_rag import RAGPipeline

        return RAGPipeline().query(payload["question"])

    def _control_mapper(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = payload.get("audit_item", "待审计对象")
        standard = payload.get("standard", "ISO27001")
        topics = payload.get("risk_topics") or ["权限", "变更", "日志"]
        controls = []
        for index, topic in enumerate(topics, start=1):
            controls.append(
                {
                    "control_id": f"MAP-{index:02d}",
                    "domain": topic,
                    "objective": f"确认{item}在{topic}领域满足{standard}相关控制要求",
                    "test_procedure": "检查制度设计、抽样验证执行记录、复核例外审批并追踪整改闭环",
                    "evidence_required": ["制度或流程文件", "审批记录", "系统配置截图", "抽样底稿", "复核记录"],
                    "quality_rule": "每项控制至少需要一项设计证据和一项运行证据，否则进入人工复核",
                }
            )
        return {"audit_item": item, "standard": standard, "control_matrix": controls}

    def _eval_case_designer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scenario = payload.get("scenario", "企业审计 Agent")
        capabilities = payload.get("capabilities") or ["RAG", "工具调用", "质量门", "人工复核"]
        cases = []
        for index, capability in enumerate(capabilities, start=1):
            cases.append(
                {
                    "case_id": f"EVAL-{index:02d}",
                    "capability": capability,
                    "question": f"在{scenario}中验证{capability}能力是否可用",
                    "expected_terms": ["证据", "来源", "风险", "结论"],
                    "pass_rule": "回答必须引用来源、给出风险判断，并说明缺失证据或人工复核条件",
                }
            )
        return {"scenario": scenario, "cases": cases, "metrics": ["retrieval_relevance", "faithfulness", "tool_success", "human_review_trigger"]}

    def _remediation_planner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        severity = payload.get("severity", "中")
        due_days = 7 if severity == "高" else 14 if severity == "中" else 30
        return {
            "title": payload.get("finding", "审计发现整改"),
            "owner_role": payload.get("owner_role", "控制责任人"),
            "due_days": due_days,
            "tasks": [
                "确认影响范围和责任人",
                "补齐控制设计和运行证据",
                "完成例外审批或权限清理",
                "由审计或内控团队复核关闭",
            ],
            "acceptance_criteria": "整改证据完整、抽样无重大例外、复核意见已记录",
            "status_flow": ["未开始", "进行中", "待验证", "已完成", "已关闭"],
        }

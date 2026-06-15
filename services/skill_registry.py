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

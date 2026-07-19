"""Persistent audit Agent runtime with A2A-style task envelopes."""

from __future__ import annotations

import json
import hashlib
import statistics
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS
from services.safety_gate import SafetyGate
from services.skill_registry import SkillRegistry


class AgentRuntime:
    """Coordinates task planning, tool execution, artifacts, and observability."""

    def __init__(self, skill_registry: SkillRegistry, safety_gate: Optional[SafetyGate] = None) -> None:
        self.skill_registry = skill_registry
        self.safety_gate = safety_gate or SafetyGate()
        self.runtime_dir = PATHS["data"] / "agent_runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = self.runtime_dir / "events.jsonl"

    def create_task(self, objective: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        task_id = f"AGT-{uuid.uuid4().hex[:10].upper()}"
        now = datetime.now().isoformat()
        plan = self._plan(objective, context)
        safety = self.safety_gate.inspect({"objective": objective, "context": context}, stage="runtime")
        status = "blocked" if safety["status"] == "blocked" else "planned"
        task = {
            "task_id": task_id,
            "protocol": "audit-agent-task-v1",
            "objective": objective,
            "context": context,
            "status": status,
            "plan": plan,
            "steps": [],
            "artifacts": [],
            "tool_calls": [],
            "reflections": [],
            "budgets": {"max_tool_calls": 10, "max_retries_per_step": 1},
            "safety_gate": safety,
            "metrics": {
                "tool_calls": 0,
                "successful_tool_calls": 0,
                "failed_tool_calls": 0,
                "avg_latency_ms": 0,
                "estimated_cost": 0,
            },
            "created_at": now,
            "updated_at": now,
        }
        self._write_task(task)
        self._append_event(task_id, "task_created", {"status": status, "plan_steps": len(plan)})
        if status != "blocked":
            self.run_next_step(task_id)
        return self.get_task(task_id) or task

    def list_tasks(self, limit: int = 30) -> List[Dict[str, Any]]:
        files = sorted(self.runtime_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        return [self._read(path) for path in files[:limit]]

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(task_id)
        if not path.exists():
            return None
        return self._read(path)

    def delete_task(self, task_id: str) -> bool:
        path = self._path(task_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def run_next_step(self, task_id: str) -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        if task["status"] == "blocked":
            return task

        max_calls = int(task.get("budgets", {}).get("max_tool_calls", 10))
        if len(task.get("tool_calls", [])) >= max_calls:
            task["status"] = "needs_review"
            task.setdefault("reflections", []).append(
                {
                    "reflection_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
                    "verdict": "human_review",
                    "confidence": 1.0,
                    "issues": ["工具调用预算已耗尽。"],
                    "next_action": "由复核人扩展预算或收窄任务范围。",
                    "created_at": datetime.now().isoformat(),
                }
            )
            self._write_task(task)
            self._append_event(task_id, "budget_exhausted", {"max_tool_calls": max_calls})
            return task

        completed = {step["step_id"] for step in task.get("steps", []) if step.get("status") == "success"}
        next_plan = next((item for item in task.get("plan", []) if item["step_id"] not in completed), None)
        if not next_plan:
            task["status"] = "completed"
            task["updated_at"] = datetime.now().isoformat()
            self._write_task(task)
            return task

        payload = self._payload_for_step(next_plan, task)
        safety = self.safety_gate.inspect(payload, stage=next_plan.get("stage", "runtime"))
        step_record = {
            "step_id": next_plan["step_id"],
            "name": next_plan["name"],
            "stage": next_plan["stage"],
            "skill": next_plan["skill"],
            "status": "pending",
            "safety_gate": safety,
            "started_at": datetime.now().isoformat(),
        }
        if safety["status"] == "blocked":
            step_record.update({"status": "blocked", "finished_at": datetime.now().isoformat(), "output": {"error": "blocked by safety gate"}})
            task["steps"].append(step_record)
            task["status"] = "blocked"
            task["updated_at"] = datetime.now().isoformat()
            self._write_task(task)
            return task

        runs = [self.skill_registry.execute(next_plan["skill"], payload)]
        retry_budget = int(task.get("budgets", {}).get("max_retries_per_step", 1))
        if runs[-1]["status"] != "success" and retry_budget > 0:
            runs.append(self.skill_registry.execute(next_plan["skill"], payload))
        run = runs[-1]
        step_record.update(
            {
                "status": run["status"],
                "finished_at": run["finished_at"],
                "run_id": run["run_id"],
                "output": run.get("output"),
                "duration_ms": run.get("duration_ms", 0),
                "attempts": len(runs),
            }
        )
        task["steps"].append(step_record)
        task.setdefault("role_traces", []).append(
            {
                "trace_id": f"ROLE-{uuid.uuid4().hex[:8].upper()}",
                "agent_role": next_plan.get("agent_role", "audit_agent"),
                "step_id": next_plan["step_id"],
                "skill": next_plan["skill"],
                "input_hash": self._hash_payload(payload),
                "decision": next_plan.get("purpose", ""),
                "status": run["status"],
                "attempts": len(runs),
                "duration_ms": run.get("duration_ms", 0),
                "artifact_refs": [],
                "handoff": {
                    "depends_on": next_plan.get("depends_on", []),
                    "next_step": self._next_step_id(task, next_plan["step_id"]),
                },
            }
        )
        for attempt, item in enumerate(runs, start=1):
            task["tool_calls"].append(
                {
                    "run_id": item["run_id"],
                    "skill": next_plan["skill"],
                    "status": item["status"],
                    "duration_ms": item.get("duration_ms", 0),
                    "input_size": item.get("input_size", 0),
                    "output_size": item.get("output_size", 0),
                    "attempt": attempt,
                    "cache_hit": item.get("cache_hit", False),
                    "circuit_state": item.get("circuit_state", "closed"),
                }
            )
        reflection = self._reflect(next_plan, run, len(runs))
        task.setdefault("reflections", []).append(reflection)
        if run["status"] == "success":
            artifacts = self._artifacts_from_run(next_plan, run)
            task["artifacts"].extend(artifacts)
            task["role_traces"][-1]["artifact_refs"] = [item["artifact_id"] for item in artifacts]
        task["metrics"] = self._metrics(task)
        if run["status"] != "success":
            task["status"] = "needs_review"
        else:
            task["status"] = "completed" if len(completed) + 1 >= len(task.get("plan", [])) else "running"
        task["updated_at"] = datetime.now().isoformat()
        self._write_task(task)
        self._append_event(
            task_id,
            "step_finished",
            {"step_id": next_plan["step_id"], "skill": next_plan["skill"], "status": run["status"], "attempts": len(runs)},
        )
        return task

    def add_step(self, task_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        task = self.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        new_step = {
            "step_id": step.get("step_id") or f"MANUAL-{len(task.get('plan', [])) + 1:02d}",
            "name": step.get("name", "Manual step"),
            "stage": step.get("stage", "manual"),
            "skill": step.get("skill", "audit.evidence_checklist"),
            "purpose": step.get("purpose", "User supplied runtime step"),
        }
        task.setdefault("plan", []).append(new_step)
        task["status"] = "planned"
        task["updated_at"] = datetime.now().isoformat()
        self._write_task(task)
        return task

    def observability(self) -> Dict[str, Any]:
        tasks = self.list_tasks(limit=200)
        tool_calls = [call for task in tasks for call in task.get("tool_calls", [])]
        latencies = [float(call.get("duration_ms") or 0) for call in tool_calls]
        success = [call for call in tool_calls if call.get("status") == "success"]
        blocked = [task for task in tasks if task.get("status") == "blocked"]
        active = [task for task in tasks if task.get("status") in {"planned", "running", "needs_review"}]
        return {
            "tasks": len(tasks),
            "active_tasks": len(active),
            "blocked_tasks": len(blocked),
            "tool_calls": len(tool_calls),
            "tool_success_rate": round(len(success) / max(len(tool_calls), 1), 3),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(statistics.quantiles(latencies, n=20)[-1], 2) if len(latencies) >= 20 else round(max(latencies), 2) if latencies else 0,
            "safety_review_rate": round(
                sum(1 for task in tasks if task.get("safety_gate", {}).get("status") == "review") / max(len(tasks), 1),
                3,
            ),
            "reflections": sum(len(task.get("reflections", [])) for task in tasks),
            "retry_count": sum(max(0, int(step.get("attempts", 1)) - 1) for task in tasks for step in task.get("steps", [])),
            "memory": {"task_checkpoints": len(tasks), "artifact_count": sum(len(task.get("artifacts", [])) for task in tasks)},
            "latest_tasks": tasks[:8],
            "skill_metrics": self.skill_registry.metrics(),
        }

    def _plan(self, objective: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
        audit_item = context.get("audit_item") or objective[:80]
        risk_topics = context.get("risk_topics") or ["权限", "变更", "日志", "数据"]
        return [
            {
                "step_id": "PLAN-01",
                "name": "审计范围规划",
                "stage": "audit",
                "skill": "audit.scope_planner",
                "agent_role": "planning_agent",
                "depends_on": [],
                "purpose": "Clarify audit scope, standard, and deliverables.",
                "input_hint": {"audit_item": audit_item, "risk_topics": risk_topics},
            },
            {
                "step_id": "MAP-02",
                "name": "控制矩阵映射",
                "stage": "audit",
                "skill": "audit.control_mapper",
                "agent_role": "control_agent",
                "depends_on": ["PLAN-01"],
                "purpose": "Map risks to executable control tests.",
                "input_hint": {"audit_item": audit_item, "risk_topics": risk_topics},
            },
            {
                "step_id": "EVD-03",
                "name": "证据清单生成",
                "stage": "evidence",
                "skill": "audit.evidence_checklist",
                "agent_role": "evidence_agent",
                "depends_on": ["PLAN-01", "MAP-02"],
                "purpose": "Generate evidence requests and collection methods.",
                "input_hint": {"control_domain": "、".join(risk_topics[:3])},
            },
            {
                "step_id": "RSH-04",
                "name": "Deep Research 研究计划",
                "stage": "research",
                "skill": "audit.deep_research_brief",
                "agent_role": "research_agent",
                "depends_on": ["PLAN-01"],
                "purpose": "Create query rewrites, source strategy, and review conditions.",
                "input_hint": {"question": objective, "standard": context.get("standard") or context.get("standard_type") or "ISO27001"},
            },
            {
                "step_id": "SMP-05",
                "name": "审计抽样方案生成",
                "stage": "evidence",
                "skill": "audit.sample_designer",
                "agent_role": "control_agent",
                "depends_on": ["MAP-02", "EVD-03"],
                "purpose": "Design sampling method without sacrificing audit confidence.",
                "input_hint": {"population": int(context.get("population") or 120), "risk_level": context.get("risk_level", "medium"), "frequency": context.get("frequency", "daily")},
            },
            {
                "step_id": "TRI-06",
                "name": "审计例外分级与处置",
                "stage": "risk",
                "skill": "audit.exception_triage",
                "agent_role": "risk_agent",
                "depends_on": ["SMP-05"],
                "purpose": "Prepare exception severity, root cause, and escalation actions.",
                "input_hint": {"finding": f"{audit_item} 控制测试例外待分级", "risk_level": context.get("risk_level", "medium")},
            },
            {
                "step_id": "REM-07",
                "name": "整改任务规划",
                "stage": "audit",
                "skill": "audit.remediation_planner",
                "agent_role": "remediation_agent",
                "depends_on": ["MAP-02", "EVD-03", "TRI-06"],
                "purpose": "Prepare remediation workflow for likely findings.",
                "input_hint": {"finding": f"{audit_item} 控制证据或执行一致性需复核", "severity": context.get("risk_level", "中")},
            },
        ]

    def _payload_for_step(self, step: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(step.get("input_hint") or {})
        payload.update(task.get("context") or {})
        payload.setdefault("objective", task.get("objective"))
        return payload

    def _artifacts_from_run(self, step: Dict[str, Any], run: Dict[str, Any]) -> List[Dict[str, Any]]:
        output = run.get("output") or {}
        return [
            {
                "artifact_id": f"ART-{uuid.uuid4().hex[:8].upper()}",
                "type": step.get("stage", "runtime"),
                "name": step.get("name"),
                "source_run_id": run.get("run_id"),
                "summary": self._summarize_output(output),
            }
        ]

    def _summarize_output(self, output: Any) -> str:
        if isinstance(output, dict):
            for key in ("scope", "title", "recommendation", "collection_method", "scenario"):
                if output.get(key):
                    return str(output[key])[:240]
            return ", ".join(output.keys())[:240]
        return str(output)[:240]

    def _reflect(self, step: Dict[str, Any], run: Dict[str, Any], attempts: int) -> Dict[str, Any]:
        output = run.get("output") or {}
        status = run.get("status")
        if status != "success":
            verdict = "human_review"
            confidence = 0.2
            issues = [str(output.get("error") or "工具执行失败")]
            next_action = "检查输入、工具依赖与熔断状态后人工决定重试或改写计划。"
        else:
            serialized = json.dumps(output, ensure_ascii=False, default=str)
            sparse = len(serialized) < 40
            verdict = "review" if sparse else "pass"
            confidence = 0.62 if sparse else min(0.96, 0.72 + len(serialized) / 6000)
            issues = ["工具输出过短，需要确认是否覆盖任务目标。"] if sparse else []
            next_action = "进入下一计划步骤。" if not sparse else "复核产物后再继续执行。"
        return {
            "reflection_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
            "step_id": step.get("step_id"),
            "agent_role": step.get("agent_role", "audit_agent"),
            "verdict": verdict,
            "confidence": round(confidence, 3),
            "issues": issues,
            "attempts": attempts,
            "next_action": next_action,
            "created_at": datetime.now().isoformat(),
        }

    def _metrics(self, task: Dict[str, Any]) -> Dict[str, Any]:
        calls = task.get("tool_calls", [])
        success = [call for call in calls if call.get("status") == "success"]
        latencies = [float(call.get("duration_ms") or 0) for call in calls]
        return {
            "tool_calls": len(calls),
            "successful_tool_calls": len(success),
            "failed_tool_calls": len(calls) - len(success),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
            "estimated_cost": 0,
            "cache_hits": sum(1 for call in calls if call.get("cache_hit")),
            "retry_count": sum(1 for call in calls if int(call.get("attempt", 1)) > 1),
        }

    def _path(self, task_id: str) -> Path:
        return self.runtime_dir / f"{task_id}.json"

    def _hash_payload(self, payload: Dict[str, Any]) -> str:
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]

    def _next_step_id(self, task: Dict[str, Any], current_step_id: str) -> Optional[str]:
        plan = task.get("plan", [])
        for index, item in enumerate(plan):
            if item.get("step_id") == current_step_id and index + 1 < len(plan):
                return plan[index + 1].get("step_id")
        return None

    def _append_event(self, task_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        record = {
            "event_id": f"AE-{uuid.uuid4().hex[:10].upper()}",
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
            "at": datetime.now().isoformat(),
        }
        with self.event_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _read(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_task(self, task: Dict[str, Any]) -> None:
        self._path(task["task_id"]).write_text(json.dumps(task, ensure_ascii=False, indent=2), encoding="utf-8")

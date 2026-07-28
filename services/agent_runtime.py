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
from services.evaluation_calibration import beta_posterior_mean, wilson_lower_bound
from services.record_store import SQLiteRecordStore
from services.safety_gate import SafetyGate
from services.skill_registry import SkillRegistry
from services.security import current_tenant_id, record_visible


class AgentRuntime:
    """Coordinates task planning, tool execution, artifacts, and observability."""

    def __init__(self, skill_registry: SkillRegistry, safety_gate: Optional[SafetyGate] = None) -> None:
        self.skill_registry = skill_registry
        self.safety_gate = safety_gate or SafetyGate()
        self.runtime_dir = PATHS["data"] / "agent_runtime"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.event_log = self.runtime_dir / "events.jsonl"
        self._record_stores: Dict[str, SQLiteRecordStore] = {}

    def create_task(self, objective: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        context = context or {}
        task_id = f"AGT-{uuid.uuid4().hex[:10].upper()}"
        now = datetime.now().isoformat()
        plan = self._plan(objective, context)
        safety = self.safety_gate.inspect({"objective": objective, "context": context}, stage="runtime")
        status = "blocked" if safety["status"] == "blocked" else "planned"
        task = {
            "task_id": task_id,
            "tenant_id": current_tenant_id(),
            "protocol": "audit-agent-task-v1",
            "objective": objective,
            "context": context,
            "applied_lessons": [
                item
                for item in context.get("experience_lessons", [])
                if isinstance(item, dict) and item.get("status") == "approved"
            ],
            "status": status,
            "plan": plan,
            "steps": [],
            "artifacts": [],
            "tool_calls": [],
            "reflections": [],
            "budgets": {"max_tool_calls": 10, "max_retries_per_step": 1},
            "loop": {
                "strategy": "bounded_dependency_loop",
                "iterations": 0,
                "termination_reason": "not_started",
                "last_started_at": None,
                "last_finished_at": None,
            },
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

    def run_until_pause(self, task_id: str, max_steps: int = 6) -> Dict[str, Any]:
        """Run a bounded loop until completion, review, blocking, or step budget."""

        task = self.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        bounded_steps = max(1, min(int(max_steps or 1), 12))
        loop = task.setdefault("loop", {})
        loop.update(
            {
                "strategy": "bounded_dependency_loop",
                "last_started_at": datetime.now().isoformat(),
                "termination_reason": "running",
            }
        )
        self._write_task(task)
        self._append_event(task_id, "loop_started", {"max_steps": bounded_steps})

        executed = 0
        while executed < bounded_steps:
            before = self.get_task(task_id) or task
            if before.get("status") in {"completed", "blocked", "needs_review"}:
                break
            before_steps = len(before.get("steps", []))
            task = self.run_next_step(task_id)
            executed += max(0, len(task.get("steps", [])) - before_steps)
            if task.get("status") in {"completed", "blocked", "needs_review"}:
                break
            if len(task.get("steps", [])) == before_steps:
                break

        task = self.get_task(task_id) or task
        if task.get("status") == "completed":
            reason = "task_completed"
        elif task.get("status") == "blocked":
            reason = "safety_blocked"
        elif task.get("status") == "needs_review":
            reason = "human_review_required"
        elif executed >= bounded_steps:
            reason = "step_budget_reached"
        else:
            reason = "no_progress"
        loop = task.setdefault("loop", {})
        loop["iterations"] = int(loop.get("iterations") or 0) + executed
        loop["last_finished_at"] = datetime.now().isoformat()
        loop["termination_reason"] = reason
        task["updated_at"] = datetime.now().isoformat()
        self._write_task(task)
        self._append_event(
            task_id,
            "loop_finished",
            {"executed_steps": executed, "termination_reason": reason, "status": task.get("status")},
        )
        return task

    def list_tasks(self, limit: int = 30) -> List[Dict[str, Any]]:
        self._migrate_legacy_tasks()
        return self._record_store().list("agent_task", limit=max(1, limit))

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._record_store().get("agent_task", task_id)
        if task is not None:
            return task
        path = self._path(task_id)
        if not path.exists():
            return None
        task = self._read(path)
        if not record_visible(task):
            return None
        task.setdefault("tenant_id", current_tenant_id())
        self._write_task(task)
        return task

    def delete_task(self, task_id: str) -> bool:
        path = self._path(task_id)
        if self.get_task(task_id) is None:
            return False
        removed = self._record_store().delete("agent_task", task_id)
        if path.exists():
            path.unlink()
        return removed

    def episode_package(self, task_id: str) -> Dict[str, Any]:
        """Build a trace-based, auditable episode without exposing raw secrets."""
        task = self.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        events = self._events_for_task(task_id)
        context = task.get("context") or {}
        reflections = task.get("reflections") or []
        failed_steps = [step for step in task.get("steps", []) if step.get("status") != "success"]
        confidence_values = [
            float(item.get("confidence"))
            for item in reflections
            if isinstance(item.get("confidence"), (int, float))
        ]
        package = {
            "schema": "audit-agent-episode-v1",
            "task_id": task_id,
            "task_specification": {
                "objective": task.get("objective"),
                "protocol": task.get("protocol"),
                "plan_steps": len(task.get("plan", [])),
                "budgets": task.get("budgets", {}),
            },
            "context_evidence": {
                "context_hash": self._hash_payload(context),
                "available_fields": sorted(context.keys()),
                "raw_context_included": False,
            },
            "action_evidence": task.get("role_traces", []),
            "tool_evidence": task.get("tool_calls", []),
            "verification_evidence": {
                "task_safety_gate": task.get("safety_gate", {}),
                "step_safety_gates": [
                    {"step_id": step.get("step_id"), "gate": step.get("safety_gate", {})}
                    for step in task.get("steps", [])
                ],
                "reflections": reflections,
            },
            "failure_attribution": [
                {
                    "step_id": step.get("step_id"),
                    "skill": step.get("skill"),
                    "status": step.get("status"),
                    "error": (step.get("output") or {}).get("error")
                    if isinstance(step.get("output"), dict)
                    else str(step.get("output") or ""),
                }
                for step in failed_steps
            ],
            "intervention_record": [
                event for event in events if event.get("event_type") in {"budget_exhausted", "manual_step_added", "human_review"}
            ],
            "entropy_audit": {
                "reflection_count": len(reflections),
                "mean_confidence": round(statistics.mean(confidence_values), 3) if confidence_values else None,
                "retry_count": task.get("metrics", {}).get("retry_count", 0),
                "cache_hits": task.get("metrics", {}).get("cache_hits", 0),
            },
            "outcome": {
                "status": task.get("status"),
                "metrics": task.get("metrics", {}),
                "artifact_refs": [item.get("artifact_id") for item in task.get("artifacts", [])],
                "completed_steps": sum(1 for step in task.get("steps", []) if step.get("status") == "success"),
            },
            "event_log": events,
            "generated_at": datetime.now().isoformat(),
        }
        package["integrity"] = {
            "algorithm": "sha256",
            "digest": self._package_digest(package),
        }
        return package

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
            task.setdefault("loop", {})["termination_reason"] = "task_completed"
            task["updated_at"] = datetime.now().isoformat()
            self._write_task(task)
            return task

        missing_dependencies = [
            dependency for dependency in next_plan.get("depends_on", []) if dependency not in completed
        ]
        if missing_dependencies:
            task["status"] = "needs_review"
            task.setdefault("reflections", []).append(
                {
                    "reflection_id": f"REF-{uuid.uuid4().hex[:8].upper()}",
                    "step_id": next_plan.get("step_id"),
                    "agent_role": next_plan.get("agent_role", "audit_agent"),
                    "verdict": "human_review",
                    "confidence": 1.0,
                    "issues": [f"计划依赖未满足：{', '.join(missing_dependencies)}"],
                    "attempts": 0,
                    "next_action": "修复计划依赖或补充前置步骤后再继续。",
                    "created_at": datetime.now().isoformat(),
                }
            )
            task.setdefault("loop", {})["termination_reason"] = "dependency_violation"
            task["updated_at"] = datetime.now().isoformat()
            self._write_task(task)
            self._append_event(
                task_id,
                "dependency_blocked",
                {"step_id": next_plan.get("step_id"), "missing_dependencies": missing_dependencies},
            )
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
            step_record["evaluation"] = self._evaluate_step(
                next_plan,
                step_record,
                task,
                completed,
            )
            task["steps"].append(step_record)
            task["status"] = "blocked"
            task.setdefault("loop", {})["termination_reason"] = "safety_blocked"
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
        step_record["evaluation"] = self._evaluate_step(
            next_plan,
            step_record,
            task,
            completed,
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
                    "error_type": item.get("error_type", ""),
                    "validation_errors": item.get("validation_errors", []),
                    "span": {
                        "name": f"execute_tool {next_plan['skill']}",
                        "kind": "execute_tool",
                        "attributes": {
                            "gen_ai.operation.name": "execute_tool",
                            "gen_ai.tool.name": next_plan["skill"],
                            "agent.task.id": task_id,
                            "agent.step.id": next_plan["step_id"],
                            "agent.role": next_plan.get("agent_role", "audit_agent"),
                            "tool.attempt": attempt,
                            "tool.cache_hit": bool(item.get("cache_hit", False)),
                            "tool.circuit_state": item.get("circuit_state", "closed"),
                        },
                    },
                }
            )
        reflection = self._reflect(next_plan, run, len(runs))
        task.setdefault("reflections", []).append(reflection)
        if run["status"] == "success":
            artifacts = self._artifacts_from_run(next_plan, run, task_id)
            task["artifacts"].extend(artifacts)
            task["role_traces"][-1]["artifact_refs"] = [item["artifact_id"] for item in artifacts]
        task["metrics"] = self._metrics(task)
        if run["status"] != "success" or step_record["evaluation"].get("status") != "pass":
            task["status"] = "needs_review"
            task.setdefault("loop", {})["termination_reason"] = (
                "tool_failure" if run["status"] != "success" else "step_evaluation_failed"
            )
        else:
            task["status"] = "completed" if len(completed) + 1 >= len(task.get("plan", [])) else "running"
            task.setdefault("loop", {})["termination_reason"] = (
                "task_completed" if task["status"] == "completed" else "step_completed"
            )
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
        self._append_event(task_id, "manual_step_added", {"step_id": new_step["step_id"], "skill": new_step["skill"]})
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
        normalized = f"{objective} {json.dumps(context, ensure_ascii=False, default=str)}".lower()
        include_research = any(
            signal in normalized
            for signal in ("标准", "法规", "iso", "sox", "cobit", "research", "研究", "外部")
        )
        include_sampling = bool(context.get("population")) or any(
            signal in normalized for signal in ("抽样", "样本", "测试", "证据", "日志")
        )
        plan = [
            {
                "step_id": "PLAN-01",
                "name": "审计范围规划",
                "stage": "audit",
                "skill": "audit.scope_planner",
                "agent_role": "planning_agent",
                "depends_on": [],
                "purpose": "Clarify audit scope, standard, and deliverables.",
                "input_hint": {"audit_item": audit_item, "risk_topics": risk_topics},
                "output_contract": ["scope", "objectives", "deliverables"],
                "termination_condition": "审计对象、范围与交付物均已明确",
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
                "output_contract": ["control_matrix", "test_procedures"],
                "termination_condition": "每个核心风险至少映射一个可执行控制",
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
                "output_contract": ["evidence_requests", "collection_methods"],
                "termination_condition": "证据请求具备来源、用途与缺口提示",
            },
        ]
        if include_research:
            plan.append(
                {
                    "step_id": "RSH-04",
                    "name": "Deep Research 研究计划",
                    "stage": "research",
                    "skill": "audit.deep_research_brief",
                    "agent_role": "research_agent",
                    "depends_on": ["PLAN-01"],
                    "purpose": "Create query rewrites, source strategy, and review conditions.",
                    "input_hint": {"question": objective, "standard": context.get("standard") or context.get("standard_type") or "ISO27001"},
                    "output_contract": ["query_rewrites", "source_strategy", "review_conditions"],
                    "termination_condition": "来源策略与人工复核条件均已形成",
                }
            )
        if include_sampling:
            plan.append(
                {
                    "step_id": "SMP-05",
                    "name": "审计抽样方案生成",
                    "stage": "evidence",
                    "skill": "audit.sample_designer",
                    "agent_role": "sampling_agent",
                    "depends_on": ["MAP-02", "EVD-03"],
                    "purpose": "Design sampling method without sacrificing audit confidence.",
                    "input_hint": {"population": int(context.get("population") or 120), "risk_level": context.get("risk_level", "medium"), "frequency": context.get("frequency", "daily")},
                    "output_contract": ["sample_size", "selection_method", "expansion_rule"],
                    "termination_condition": "样本规模、选择方法与例外扩样规则均已明确",
                }
            )
        triage_dependencies = ["SMP-05"] if include_sampling else ["MAP-02", "EVD-03"]
        plan.extend(
            [
            {
                "step_id": "TRI-06",
                "name": "审计例外分级与处置",
                "stage": "risk",
                "skill": "audit.exception_triage",
                "agent_role": "risk_agent",
                "depends_on": triage_dependencies,
                "purpose": "Prepare exception severity, root cause, and escalation actions.",
                "input_hint": {"finding": f"{audit_item} 控制测试例外待分级", "risk_level": context.get("risk_level", "medium")},
                "output_contract": ["severity", "root_cause", "escalation"],
                "termination_condition": "例外已分级并给出升级处置路径",
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
                "output_contract": ["owner", "due_date", "acceptance_criteria"],
                "termination_condition": "整改责任、期限与验收标准均已明确",
            },
            ]
        )
        verification_dependencies = [item["step_id"] for item in plan if item["step_id"] != "PLAN-01"]
        plan.extend(
            [
                {
                    "step_id": "VER-08",
                    "name": "独立交付验证",
                    "stage": "verification",
                    "skill": "audit.delivery_verifier",
                    "agent_role": "verification_agent",
                    "depends_on": verification_dependencies,
                    "purpose": "Independently verify upstream artifacts, provenance and delivery completeness.",
                    "input_hint": {"audit_item": audit_item},
                    "output_contract": ["verdict", "checks", "next_action"],
                    "termination_condition": "验证结论为 pass 或明确转人工复核",
                },
                {
                    "step_id": "PKG-09",
                    "name": "审计报告交付打包",
                    "stage": "delivery",
                    "skill": "audit.report_packager",
                    "agent_role": "delivery_agent",
                    "depends_on": ["VER-08"],
                    "purpose": "Package verified artifacts into an auditable delivery structure.",
                    "input_hint": {"audit_item": audit_item},
                    "output_contract": ["delivery_manifest", "review_status"],
                    "termination_condition": "交付目录与复核状态均已形成",
                },
            ]
        )
        return plan

    def _payload_for_step(self, step: Dict[str, Any], task: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(step.get("input_hint") or {})
        payload.update(task.get("context") or {})
        payload.setdefault("objective", task.get("objective"))
        payload.setdefault(
            "upstream_artifacts",
            json.loads(json.dumps(task.get("artifacts", []), ensure_ascii=False, default=str)),
        )
        payload.setdefault(
            "upstream_steps",
            [
                {
                    "step_id": item.get("step_id"),
                    "status": item.get("status"),
                    "skill": item.get("skill"),
                    "evaluation": item.get("evaluation", {}),
                }
                for item in task.get("steps", [])
            ],
        )
        return payload

    def _artifacts_from_run(
        self,
        step: Dict[str, Any],
        run: Dict[str, Any],
        task_id: str,
    ) -> List[Dict[str, Any]]:
        output = json.loads(json.dumps(run.get("output") or {}, ensure_ascii=False, default=str))
        canonical = json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return [
            {
                "artifact_id": f"ART-{uuid.uuid4().hex[:8].upper()}",
                "type": step.get("stage", "runtime"),
                "name": step.get("name"),
                "source_run_id": run.get("run_id"),
                "summary": self._summarize_output(output),
                "content": output,
                "content_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
                "schema_version": f"{step.get('skill', 'runtime')}:1",
                "producer": {
                    "task_id": task_id,
                    "step_id": step.get("step_id"),
                    "skill": step.get("skill"),
                    "agent_role": step.get("agent_role"),
                },
                "created_at": datetime.now().isoformat(),
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

    def _evaluate_step(
        self,
        plan_step: Dict[str, Any],
        step_record: Dict[str, Any],
        task: Dict[str, Any],
        completed: set[str],
    ) -> Dict[str, Any]:
        """Evaluate a single execution step before the loop can continue."""

        dependencies = plan_step.get("depends_on", [])
        assertions = [
            {
                "key": "dependency_conformance",
                "label": "前置依赖已完成",
                "passed": all(item in completed for item in dependencies),
                "evidence": dependencies,
            },
            {
                "key": "safety_gate",
                "label": "步骤安全门允许执行",
                "passed": step_record.get("safety_gate", {}).get("status") != "blocked",
                "evidence": step_record.get("safety_gate", {}).get("status"),
            },
            {
                "key": "tool_result",
                "label": "工具返回结构化结果",
                "passed": step_record.get("status") == "success"
                and isinstance(step_record.get("output"), dict),
                "evidence": step_record.get("status"),
            },
            {
                "key": "retry_budget",
                "label": "重试次数未超预算",
                "passed": int(step_record.get("attempts") or 0)
                <= int(task.get("budgets", {}).get("max_retries_per_step", 1)) + 1,
                "evidence": step_record.get("attempts"),
            },
            {
                "key": "provenance",
                "label": "运行记录具备来源标识",
                "passed": bool(step_record.get("run_id")) or step_record.get("status") == "blocked",
                "evidence": step_record.get("run_id"),
            },
            {
                "key": "output_contract",
                "label": "输出契约字段完整且非空",
                "passed": self._contract_satisfied(
                    step_record.get("output"),
                    plan_step.get("output_contract", []),
                ),
                "evidence": plan_step.get("output_contract", []),
            },
            {
                "key": "termination_condition",
                "label": "步骤具备明确终止条件并已满足",
                "passed": bool(plan_step.get("termination_condition"))
                and self._contract_satisfied(
                    step_record.get("output"),
                    plan_step.get("output_contract", []),
                ),
                "evidence": plan_step.get("termination_condition"),
            },
        ]
        passed = sum(1 for assertion in assertions if assertion["passed"])
        score = beta_posterior_mean(passed, len(assertions))
        critical_passed = all(
            assertion["passed"]
            for assertion in assertions
            if assertion["key"]
            in {
                "dependency_conformance",
                "safety_gate",
                "tool_result",
                "output_contract",
                "termination_condition",
            }
        )
        return {
            "evaluator": "online_step_contract_v2",
            "score": score,
            "raw_pass_rate": round(passed / len(assertions), 4),
            "confidence_lower_bound": wilson_lower_bound(passed, len(assertions)),
            "status": "pass" if critical_passed and score >= 0.8 else "review" if score >= 0.6 else "blocked",
            "assertions": assertions,
            "evaluated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _contract_satisfied(output: Any, required_fields: List[str]) -> bool:
        if not isinstance(output, dict) or not required_fields:
            return False
        return all(
            field in output and output[field] not in (None, "", [], {})
            for field in required_fields
        )

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
            "tenant_id": current_tenant_id(),
            "task_id": task_id,
            "event_type": event_type,
            "payload": payload,
            "at": datetime.now().isoformat(),
        }
        self._record_store().put("agent_event", record["event_id"], record)

    def _events_for_task(self, task_id: str) -> List[Dict[str, Any]]:
        self._migrate_legacy_events()
        events = [
            event
            for event in self._record_store().list("agent_event", limit=5000)
            if event.get("task_id") == task_id
        ]
        return sorted(events, key=lambda item: item.get("at") or "")

    def _package_digest(self, package: Dict[str, Any]) -> str:
        digest_input = {key: value for key, value in package.items() if key != "integrity"}
        serialized = json.dumps(digest_input, ensure_ascii=False, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _read(self, path: Path) -> Dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_task(self, task: Dict[str, Any]) -> None:
        task.setdefault("tenant_id", current_tenant_id())
        expected = task.get("_storage_version")
        version = self._record_store().put(
            "agent_task",
            str(task["task_id"]),
            task,
            expected_version=int(expected) if expected is not None else None,
        )
        task["_storage_version"] = version

    def _record_store(self) -> SQLiteRecordStore:
        key = str(self.runtime_dir.resolve())
        if key not in self._record_stores:
            self.runtime_dir.mkdir(parents=True, exist_ok=True)
            self._record_stores[key] = SQLiteRecordStore(self.runtime_dir / ".records.sqlite3")
        return self._record_stores[key]

    def _migrate_legacy_tasks(self) -> None:
        store = self._record_store()
        for path in self.runtime_dir.glob("AGT-*.json"):
            try:
                task = self._read(path)
            except (OSError, json.JSONDecodeError):
                continue
            task_id = str(task.get("task_id") or path.stem)
            if record_visible(task) and store.get("agent_task", task_id) is None:
                task.setdefault("tenant_id", current_tenant_id())
                store.put("agent_task", task_id, task)

    def _migrate_legacy_events(self) -> None:
        event_log = self.runtime_dir / "events.jsonl"
        if not event_log.exists():
            return
        store = self._record_store()
        for line in event_log.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_id = str(event.get("event_id") or "")
            if event_id and record_visible(event) and store.get("agent_event", event_id) is None:
                event.setdefault("tenant_id", current_tenant_id())
                store.put("agent_event", event_id, event)

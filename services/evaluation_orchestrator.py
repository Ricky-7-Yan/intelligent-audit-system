"""Trace-native, component-level evaluation for audit agent tasks."""

from __future__ import annotations

import math
import statistics
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from services.component_contracts import component_catalog
from services.evaluation_calibration import beta_posterior_mean, wilson_lower_bound


class ComponentEvaluationOrchestrator:
    """Evaluate outcome, trajectory, tools, evidence, safety, and delivery.

    The evaluator is deterministic by default so it can run in CI and release
    gates without a model key.  Model-based judges can be layered on top of the
    persisted report, but they never replace the executable assertions here.
    """

    VERSION = "4.0.0"
    PRIOR_ALPHA = 3.0
    PRIOR_BETA = 1.0

    def __init__(self, runtime, repository, skill_registry) -> None:
        self.runtime = runtime
        self.repository = repository
        self.skill_registry = skill_registry

    def catalog(self) -> Dict[str, Any]:
        components = component_catalog()
        return {
            "version": self.VERSION,
            "evaluation_unit": "task_trial_dimension_and_component",
            "components": components,
            "principles": [
                "分别评测最终结果、完整轨迹、单步决策、工具调用、证据质量和安全边界。",
                "聚合分采用贝叶斯平滑并报告置信下界，避免少量样本产生虚假的满分。",
                "确定性断言负责可验证事实，语义评审和人工复核只补充开放性质量判断。",
                "关键安全断言失败一票否决；单任务只能进入人工复核，不能直接视为可发布。",
            ],
        }

    def evaluate_task(self, task_id: str, persist: bool = True) -> Dict[str, Any]:
        task = self.runtime.get_task(task_id)
        if not task:
            raise KeyError(task_id)
        episode = self.runtime.episode_package(task_id)
        graph = self._build_evidence_graph(task, episode)
        components = self._evaluate_components(task, episode, graph)
        weighted_score = round(
            sum(float(item["score"]) * float(item["weight"]) for item in components)
            / max(sum(float(item["weight"]) for item in components), 0.001),
            4,
        )
        assertion_count = sum(len(item["assertions"]) for item in components)
        passed_assertions = sum(
            1 for item in components for assertion in item["assertions"] if assertion["passed"]
        )
        critical_failures = [
            f"{item['name']} / {assertion['label']}"
            for item in components
            if item.get("critical")
            for assertion in item["assertions"]
            if not assertion["passed"] and assertion.get("critical", True)
        ]
        raw_pass_rate = round(passed_assertions / max(assertion_count, 1), 4)
        pass_rate = self._calibrated_rate(passed_assertions, assertion_count)
        confidence_lower_bound = self._wilson_lower_bound(passed_assertions, assertion_count)
        evidence_count = sum(
            1
            for item in components
            for assertion in item["assertions"]
            if self._has_evidence(assertion.get("evidence"))
        )
        evidence_coverage = self._calibrated_rate(evidence_count, assertion_count, alpha=2.0, beta=1.0)
        sample_adequacy = round(min(0.999, 1 - math.exp(-assertion_count / 36)), 4)
        release_gate = self._release_gate(
            weighted_score,
            pass_rate,
            critical_failures,
            confidence_lower_bound=confidence_lower_bound,
            evidence_coverage=evidence_coverage,
            trial_count=1,
        )
        report = {
            "schema": "audit-agent-evaluation-v2",
            "evaluator_version": self.VERSION,
            "task_id": task_id,
            "task_status": task.get("status"),
            "summary": {
                "overall_score": weighted_score,
                "pass_rate": pass_rate,
                "raw_pass_rate": raw_pass_rate,
                "confidence_lower_bound": confidence_lower_bound,
                "evidence_coverage": evidence_coverage,
                "sample_adequacy": sample_adequacy,
                "trial_count": 1,
                "component_count": len(components),
                "assertion_count": assertion_count,
                "passed_assertions": passed_assertions,
                "critical_failures": critical_failures,
                "avg_latency_ms": task.get("metrics", {}).get("avg_latency_ms", 0),
            },
            "dimensions": self._public_dimensions(components),
            "components": components,
            "evidence_graph": graph,
            "release_gate": release_gate,
            "trace_binding": {
                "episode_schema": episode.get("schema"),
                "episode_digest": episode.get("integrity", {}).get("digest"),
                "raw_context_included": episode.get("context_evidence", {}).get("raw_context_included"),
            },
            "grader_profile": {
                "active": ["deterministic_assertions", "trace_and_outcome_checks"],
                "available_on_review": ["rubric_model_judge", "human_auditor"],
                "aggregation": "beta_posterior_mean_with_wilson_lower_bound",
            },
            "evaluated_at": datetime.now().isoformat(),
        }
        if persist:
            run = self.repository.create_run(
                "task_component",
                {"task_id": task_id, "case_count": 1, "source": "server_trace"},
                report,
            )
            report["evaluation_run_id"] = run["run_id"]
            report["release_gate"] = run["release_gate"]
        return report

    def evaluate_runtime(self, limit: int = 12, persist: bool = True) -> Dict[str, Any]:
        tasks = self.runtime.list_tasks(limit=max(1, min(limit, 30)))
        reports = [self.evaluate_task(task["task_id"], persist=False) for task in tasks]
        scores = [float(item["summary"]["overall_score"]) for item in reports]
        passed_assertions = sum(int(item["summary"]["passed_assertions"]) for item in reports)
        assertion_count = sum(int(item["summary"]["assertion_count"]) for item in reports)
        component_scores: Dict[str, List[float]] = {}
        for report in reports:
            for component in report["components"]:
                component_scores.setdefault(component["id"], []).append(float(component["score"]))
        weakest = sorted(
            (
                {
                    "component_id": component_id,
                    "score": round(statistics.mean(values), 4),
                }
                for component_id, values in component_scores.items()
            ),
            key=lambda item: item["score"],
        )
        contract_map = {item["id"]: item for item in component_catalog()}
        aggregated_components = [
            {
                "id": item["component_id"],
                "name": contract_map[item["component_id"]]["name"],
                "owner": contract_map[item["component_id"]]["owner"],
                "score": item["score"],
                "status": "pass" if item["score"] >= 0.8 else "review" if item["score"] >= 0.6 else "blocked",
                "task_count": len(component_scores[item["component_id"]]),
                "assertions": [],
            }
            for item in sorted(weakest, key=lambda value: value["component_id"])
        ]
        critical_failures = [
            f"{report['task_id']}: {failure}"
            for report in reports
            for failure in report["summary"]["critical_failures"]
        ]
        overall = round(statistics.mean(scores), 4) if scores else 0.0
        pass_rate = self._calibrated_rate(passed_assertions, assertion_count)
        raw_pass_rate = round(passed_assertions / max(assertion_count, 1), 4)
        confidence_lower_bound = self._wilson_lower_bound(passed_assertions, assertion_count)
        evidence_assertions = sum(
            1
            for report in reports
            for component in report["components"]
            for assertion in component["assertions"]
            if self._has_evidence(assertion.get("evidence"))
        )
        evidence_coverage = self._calibrated_rate(
            evidence_assertions,
            assertion_count,
            alpha=2.0,
            beta=1.0,
        )
        result = {
            "schema": "audit-agent-suite-evaluation-v2",
            "evaluator_version": self.VERSION,
            "summary": {
                "overall_score": overall,
                "pass_rate": pass_rate,
                "raw_pass_rate": raw_pass_rate,
                "confidence_lower_bound": confidence_lower_bound,
                "evidence_coverage": evidence_coverage,
                "sample_adequacy": round(min(0.999, 1 - math.exp(-assertion_count / 36)), 4),
                "assertion_count": assertion_count,
                "passed_assertions": passed_assertions,
                "trial_count": len(reports),
                "total_tests": len(reports),
                "component_count": len(component_scores),
                "critical_failures": critical_failures,
                "avg_latency_ms": round(
                    statistics.mean(
                        float(report["summary"].get("avg_latency_ms") or 0) for report in reports
                    ),
                    2,
                )
                if reports
                else 0,
            },
            "weakest_components": weakest[:4],
            "dimensions": self._public_dimensions(aggregated_components),
            "components": aggregated_components,
            "task_reports": reports,
            "release_gate": self._release_gate(
                overall,
                pass_rate,
                critical_failures,
                confidence_lower_bound=confidence_lower_bound,
                evidence_coverage=evidence_coverage,
                trial_count=len(reports),
            ),
            "evaluated_at": datetime.now().isoformat(),
        }
        if persist:
            run = self.repository.create_run(
                "runtime_component",
                {"case_count": len(reports), "source": "server_traces"},
                result,
            )
            result["evaluation_run_id"] = run["run_id"]
            result["release_gate"] = run["release_gate"]
        return result

    def _evaluate_components(
        self,
        task: Dict[str, Any],
        episode: Dict[str, Any],
        graph: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        contracts = {item["id"]: item for item in component_catalog()}
        plan = task.get("plan", [])
        steps = task.get("steps", [])
        calls = task.get("tool_calls", [])
        artifacts = task.get("artifacts", [])
        reflections = task.get("reflections", [])
        completed = {step.get("step_id") for step in steps if step.get("status") == "success"}
        dependency_ok = all(
            dependency in completed
            for step in steps
            for dependency in next(
                (item.get("depends_on", []) for item in plan if item.get("step_id") == step.get("step_id")),
                [],
            )
        )
        blocked_executed = any(
            step.get("safety_gate", {}).get("status") == "blocked" and step.get("run_id")
            for step in steps
        )
        online_evaluations = [step.get("evaluation") for step in steps if step.get("evaluation")]

        definitions: Dict[str, List[Dict[str, Any]]] = {
            "task_specification": [
                self._assert("objective", "目标已定义", bool(str(task.get("objective") or "").strip()), task.get("objective")),
                self._assert("budget", "运行预算有上限", int(task.get("budgets", {}).get("max_tool_calls", 0)) > 0, task.get("budgets")),
                self._assert("review_exit", "具备人工复核出口", any("review" in str(item.get("verdict", "")) for item in reflections) or task.get("status") != "blocked", task.get("status")),
            ],
            "agent_loop": [
                self._assert("plan", "计划包含可执行步骤", bool(plan), len(plan)),
                self._assert("dependencies", "已执行步骤遵守依赖", dependency_ok, sorted(completed)),
                self._assert("bounded", "工具调用未超预算", len(calls) <= int(task.get("budgets", {}).get("max_tool_calls", 0) or 0), len(calls)),
                self._assert("termination", "循环具备明确状态", task.get("status") in {"planned", "running", "completed", "needs_review", "blocked"}, task.get("status")),
            ],
            "tool_runtime": [
                self._assert("trace", "工具调用已记录", bool(calls) or not steps, len(calls)),
                self._assert("schema", "工具输入校验无失败", not any(call.get("error_type") == "InputValidationError" for call in calls), [call.get("run_id") for call in calls]),
                self._assert("success", "工具成功率达到 80%", self._ratio(call.get("status") == "success" for call in calls) >= 0.8 if calls else True, self._ratio(call.get("status") == "success" for call in calls)),
                self._assert("spans", "工具轨迹具备标准语义字段", all(call.get("span", {}).get("name") for call in calls), len(calls)),
            ],
            "evidence_grounding": [
                self._assert("evidence_step", "计划包含取证或研究步骤", any(item.get("stage") in {"evidence", "research"} for item in plan), [item.get("stage") for item in plan]),
                self._assert("evidence_artifact", "已执行取证步骤形成产物", not any(step.get("stage") in {"evidence", "research"} and step.get("status") == "success" for step in steps) or any(item.get("type") in {"evidence", "research"} for item in artifacts), [item.get("type") for item in artifacts]),
                self._assert("gap_visibility", "反思会显式保留问题", all("issues" in item for item in reflections), len(reflections)),
            ],
            "evidence_graph": [
                self._assert("provenance", "产物来源覆盖率达到 95%", graph["metrics"]["provenance_coverage"] >= 0.95, graph["metrics"]["provenance_coverage"]),
                self._assert("dependencies", "不存在断裂依赖", graph["metrics"]["broken_dependencies"] == 0, graph["metrics"]["broken_dependencies"]),
                self._assert("orphans", "关键节点孤点率低于 10%", graph["metrics"]["orphan_rate"] <= 0.1, graph["metrics"]["orphan_rate"]),
            ],
            "safety_governance": [
                self._assert("task_gate", "任务安全门已执行", bool(task.get("safety_gate", {}).get("status")), task.get("safety_gate", {}).get("status")),
                self._assert("blocked_action", "阻断步骤未调用工具", not blocked_executed, blocked_executed, critical=True),
                self._assert("redaction", "评测包不包含原始上下文", episode.get("context_evidence", {}).get("raw_context_included") is False, episode.get("context_evidence")),
            ],
            "memory_context": [
                self._assert("checkpoint", "任务检查点已持久化", bool(task.get("updated_at")), task.get("updated_at")),
                self._assert("events", "关键运行事件已记录", bool(episode.get("event_log")), len(episode.get("event_log", []))),
                self._assert("lessons", "仅使用已批准经验", all(item.get("status") == "approved" for item in task.get("applied_lessons", [])), [item.get("experience_id") for item in task.get("applied_lessons", [])]),
            ],
            "audit_delivery": [
                self._assert("artifacts", "成功步骤产生可追溯产物", len(artifacts) >= len(completed), {"artifacts": len(artifacts), "completed": len(completed)}),
                self._assert("remediation", "计划包含整改闭环", any(item.get("skill") == "audit.remediation_planner" for item in plan), [item.get("skill") for item in plan]),
                self._assert("final_state", "最终交付不会跳过复核状态", task.get("status") != "completed" or len(completed) == len(plan), task.get("status")),
            ],
            "improvement_governance": [
                self._assert("reflection", "已执行步骤均有反思", len(reflections) >= len(steps), {"reflections": len(reflections), "steps": len(steps)}),
                self._assert("online_eval", "已执行步骤均有单步评测", len(online_evaluations) >= len(steps), {"evaluated": len(online_evaluations), "steps": len(steps)}),
                self._assert("attribution", "失败步骤具备归因", all(item.get("error") for item in episode.get("failure_attribution", [])), len(episode.get("failure_attribution", []))),
            ],
        }

        results: List[Dict[str, Any]] = []
        for component_id, assertions in definitions.items():
            contract = contracts[component_id]
            passed = sum(1 for item in assertions if item["passed"])
            score = self._calibrated_rate(passed, len(assertions))
            critical_failed = any(
                not item["passed"] and item.get("critical", True) for item in assertions
            )
            results.append(
                {
                    "id": component_id,
                    "name": contract["name"],
                    "owner": contract["owner"],
                    "score": score,
                    "raw_pass_rate": round(passed / max(len(assertions), 1), 4),
                    "confidence_lower_bound": self._wilson_lower_bound(passed, len(assertions)),
                    "evidence_coverage": self._calibrated_rate(
                        sum(1 for item in assertions if self._has_evidence(item.get("evidence"))),
                        len(assertions),
                        alpha=2.0,
                        beta=1.0,
                    ),
                    "status": (
                        "blocked"
                        if critical_failed
                        else "pass"
                        if score >= 0.82
                        else "review"
                    ),
                    "weight": contract["weight"],
                    "critical": contract["critical"],
                    "assertions": assertions,
                }
            )
        return results

    def _build_evidence_graph(self, task: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
        nodes: List[Dict[str, Any]] = [
            {"id": task["task_id"], "type": "task", "label": str(task.get("objective") or "")[:80]}
        ]
        edges: List[Dict[str, Any]] = []
        node_ids = {task["task_id"]}
        for plan in task.get("plan", []):
            node_id = str(plan.get("step_id"))
            nodes.append({"id": node_id, "type": "plan_step", "label": plan.get("name")})
            node_ids.add(node_id)
            edges.append({"from": task["task_id"], "to": node_id, "type": "plans"})
        for plan in task.get("plan", []):
            for dependency in plan.get("depends_on", []):
                edges.append({"from": dependency, "to": plan.get("step_id"), "type": "depends_on"})
        for step in task.get("steps", []):
            run_id = step.get("run_id")
            if run_id:
                nodes.append({"id": run_id, "type": "tool_run", "label": step.get("skill")})
                node_ids.add(run_id)
                edges.append({"from": step.get("step_id"), "to": run_id, "type": "executes"})
        for artifact in task.get("artifacts", []):
            artifact_id = artifact.get("artifact_id")
            nodes.append({"id": artifact_id, "type": "artifact", "label": artifact.get("name")})
            node_ids.add(artifact_id)
            if artifact.get("source_run_id"):
                edges.append({"from": artifact.get("source_run_id"), "to": artifact_id, "type": "produces"})
        for reflection in task.get("reflections", []):
            reflection_id = reflection.get("reflection_id")
            nodes.append({"id": reflection_id, "type": "evaluation", "label": reflection.get("verdict")})
            node_ids.add(reflection_id)
            source = reflection.get("step_id") or task["task_id"]
            edges.append({"from": source, "to": reflection_id, "type": "evaluates"})

        broken_dependencies = sum(
            1
            for edge in edges
            if edge["type"] == "depends_on" and (edge["from"] not in node_ids or edge["to"] not in node_ids)
        )
        artifacts = task.get("artifacts", [])
        provenance_coverage = self._ratio(bool(item.get("source_run_id")) for item in artifacts) if artifacts else 1.0
        connected = {edge["from"] for edge in edges} | {edge["to"] for edge in edges}
        critical_nodes = [node for node in nodes if node["type"] in {"plan_step", "artifact", "tool_run"}]
        orphan_rate = round(
            sum(1 for node in critical_nodes if node["id"] not in connected) / max(len(critical_nodes), 1),
            3,
        )
        return {
            "schema": "audit-evidence-graph-v1",
            "nodes": nodes,
            "edges": edges,
            "metrics": {
                "node_count": len(nodes),
                "edge_count": len(edges),
                "provenance_coverage": provenance_coverage,
                "broken_dependencies": broken_dependencies,
                "orphan_rate": orphan_rate,
                "episode_digest": episode.get("integrity", {}).get("digest"),
            },
        }

    def _release_gate(
        self,
        score: float,
        pass_rate: float,
        critical_failures: List[str],
        *,
        confidence_lower_bound: float,
        evidence_coverage: float,
        trial_count: int,
    ) -> Dict[str, Any]:
        blockers: List[str] = []
        review_reasons: List[str] = []
        if score < 0.70:
            blockers.append("校准总分低于 70%。")
        elif score < 0.82:
            review_reasons.append("校准总分尚未达到 82% 的发布候选线。")
        if pass_rate < 0.72:
            blockers.append("校准通过率低于 72%。")
        if evidence_coverage < 0.72:
            blockers.append("可验证证据覆盖率低于 72%。")
        if confidence_lower_bound < 0.68:
            review_reasons.append("95% 置信下界低于 68%，需要扩大样本或人工复核。")
        if trial_count < 3:
            review_reasons.append("独立试验少于 3 次，单次任务不能直接作为发布结论。")
        blockers.extend(critical_failures)
        status = "blocked" if blockers else "review" if review_reasons else "pass"
        return {
            "status": status,
            "label": {"pass": "可进入发布复核", "review": "需人工复核", "blocked": "阻断发布"}[status],
            "blockers": blockers,
            "review_reasons": review_reasons,
            "thresholds": {
                "overall_score": 0.82,
                "pass_rate": 0.72,
                "confidence_lower_bound": 0.68,
                "evidence_coverage": 0.72,
                "minimum_trials": 3,
                "critical_failures": 0,
            },
        }

    def _public_dimensions(self, components: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expose customer-facing quality dimensions, not internal ownership boundaries."""
        definitions = [
            ("task_outcome", "任务结果", ["task_specification", "audit_delivery"]),
            ("trajectory", "规划与轨迹", ["agent_loop"]),
            ("tool_use", "工具调用", ["tool_runtime"]),
            ("grounding", "证据与溯源", ["evidence_grounding", "evidence_graph"]),
            ("safety", "安全与权限", ["safety_governance"]),
            ("context", "上下文与记忆", ["memory_context"]),
            ("robustness", "复盘与改进", ["improvement_governance"]),
        ]
        by_id = {item["id"]: item for item in components}
        dimensions: List[Dict[str, Any]] = []
        for dimension_id, name, component_ids in definitions:
            selected = [by_id[item_id] for item_id in component_ids if item_id in by_id]
            if not selected:
                continue
            score = round(statistics.mean(float(item["score"]) for item in selected), 4)
            confidence_values = [
                float(item["confidence_lower_bound"])
                for item in selected
                if item.get("confidence_lower_bound") is not None
            ]
            dimensions.append(
                {
                    "id": dimension_id,
                    "name": name,
                    "score": score,
                    "confidence_lower_bound": (
                        round(min(confidence_values), 4) if confidence_values else None
                    ),
                    "status": (
                        "blocked"
                        if any(item.get("status") == "blocked" for item in selected)
                        else "pass"
                        if score >= 0.82
                        else "review"
                    ),
                    "signal_count": sum(len(item.get("assertions", [])) for item in selected),
                }
            )
        return dimensions

    def _calibrated_rate(
        self,
        successes: int,
        total: int,
        *,
        alpha: float | None = None,
        beta: float | None = None,
    ) -> float:
        prior_alpha = self.PRIOR_ALPHA if alpha is None else alpha
        prior_beta = self.PRIOR_BETA if beta is None else beta
        return beta_posterior_mean(
            successes,
            total,
            alpha=prior_alpha,
            beta=prior_beta,
        )

    def _wilson_lower_bound(self, successes: int, total: int, z: float = 1.96) -> float:
        return wilson_lower_bound(successes, total, z=z)

    def _has_evidence(self, evidence: Any) -> bool:
        if evidence is None:
            return False
        if isinstance(evidence, str):
            return bool(evidence.strip())
        if isinstance(evidence, (list, tuple, set, dict)):
            return bool(evidence)
        return True

    def _assert(
        self,
        key: str,
        label: str,
        passed: bool,
        evidence: Any,
        critical: bool = True,
    ) -> Dict[str, Any]:
        return {
            "key": key,
            "label": label,
            "passed": bool(passed),
            "score": 1.0 if passed else 0.0,
            "critical": critical,
            "evidence": evidence,
        }

    def _ratio(self, values: Iterable[bool]) -> float:
        items = list(values)
        return round(sum(1 for value in items if value) / max(len(items), 1), 3)

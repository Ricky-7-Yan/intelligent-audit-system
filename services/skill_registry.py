"""Agent skill registry and MCP-style tool facade."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Tuple

from config import PATHS
from services.security import current_principal, current_request_id, has_permission, record_visible


@dataclass
class Skill:
    name: str
    title: str
    description: str
    input_schema: Dict[str, Any]
    permissions: List[str]
    handler: Callable[[Dict[str, Any]], Dict[str, Any]]
    output_schema: Dict[str, Any] | None = None
    version: str = "1.1.0"
    timeout_seconds: float = 8.0
    cache_ttl_seconds: int = 0
    failure_threshold: int = 3


class SkillRegistry:
    def __init__(self) -> None:
        self.skills: Dict[str, Skill] = {}
        self.log_file = PATHS["data"] / "skill_runs" / "runs.jsonl"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self._executor = ThreadPoolExecutor(max_workers=6, thread_name_prefix="audit-skill")
        self._cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._circuits: Dict[str, Dict[str, Any]] = {}
        self._log_lock = threading.RLock()
        self._register_builtin_skills()

    def list_skills(self) -> List[Dict[str, Any]]:
        return [self._describe(skill) for skill in self.skills.values()]

    def mcp_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": skill.name,
                "qualifiedName": f"AuditPilot:{skill.name}",
                "description": skill.description,
                "inputSchema": skill.input_schema,
                "outputSchema": {
                    "type": "object",
                    "properties": {
                        "ok": {"type": "boolean"},
                        "data": skill.output_schema or {"type": "object"},
                        "error": {"type": "object"},
                        "meta": {"type": "object"},
                    },
                },
                "annotations": {
                    "title": skill.title,
                    "permissions": skill.permissions,
                    "version": skill.version,
                    "timeoutSeconds": skill.timeout_seconds,
                    "cacheTtlSeconds": skill.cache_ttl_seconds,
                    "failureThreshold": skill.failure_threshold,
                },
            }
            for skill in self.skills.values()
        ]

    def execute(
        self,
        name: str,
        payload: Dict[str, Any],
        permission_context: List[str] | None = None,
    ) -> Dict[str, Any]:
        if name not in self.skills:
            raise KeyError(name)
        skill = self.skills[name]
        run_id = f"SK-{uuid.uuid4().hex[:10].upper()}"
        started = datetime.now().isoformat()
        start_monotonic = time.perf_counter()
        error_type = ""
        cache_hit = False
        validation_errors = self._validate(payload, skill.input_schema)
        principal = current_principal()
        missing_permissions = [
            permission
            for permission in skill.permissions
            if not self._permission_allowed(permission, permission_context)
        ]
        circuit = self._circuits.setdefault(name, {"failures": 0, "state": "closed", "open_until": 0.0})
        cache_key = self._cache_key(name, payload)

        if missing_permissions:
            result = self._error_payload(
                "PERMISSION_DENIED",
                "当前身份没有执行该工具所需权限",
                False,
                "联系审计项目管理员授予最小必要权限。",
                {"required": skill.permissions, "missing": missing_permissions},
            )
            status = "failed"
            error_type = "PermissionDeniedError"
        elif validation_errors:
            result = self._error_payload(
                "INPUT_VALIDATION_ERROR",
                "输入校验失败",
                False,
                "按 inputSchema 补充必填字段并修正字段类型后重试。",
                validation_errors,
            )
            status = "failed"
            error_type = "InputValidationError"
        elif circuit["state"] == "open" and time.time() < float(circuit["open_until"]):
            retry_after = round(float(circuit["open_until"]) - time.time(), 2)
            result = self._error_payload(
                "CIRCUIT_OPEN",
                "工具熔断器处于开启状态",
                True,
                f"等待 {retry_after} 秒后重试，或选择等价的降级工具。",
                {"retry_after": retry_after},
            )
            status = "failed"
            error_type = "CircuitOpenError"
        elif self._cached(cache_key):
            result = self._cache[cache_key][1]
            status = "success"
            cache_hit = True
        else:
            if circuit["state"] == "open":
                circuit["state"] = "half_open"
        try:
            if not missing_permissions and not validation_errors and not cache_hit and error_type != "CircuitOpenError":
                future = self._executor.submit(skill.handler, payload)
                result = future.result(timeout=skill.timeout_seconds)
                if not isinstance(result, dict):
                    result = {"value": result}
                output_validation_errors = self._validate(result, skill.output_schema or {})
                if output_validation_errors:
                    result = self._error_payload(
                        "OUTPUT_VALIDATION_ERROR",
                        "工具输出不符合声明的 outputSchema",
                        False,
                        "修复工具实现或升级输出契约后再执行；禁止把不完整产物交给下游 Agent。",
                        output_validation_errors,
                    )
                    status = "failed"
                    error_type = "OutputValidationError"
                    self._record_failure(skill, circuit)
                else:
                    status = "success"
                    circuit.update({"failures": 0, "state": "closed", "open_until": 0.0})
                    if skill.cache_ttl_seconds > 0:
                        self._cache[cache_key] = (time.time() + skill.cache_ttl_seconds, result)
        except FutureTimeoutError:
            result = self._error_payload(
                "TOOL_TIMEOUT",
                f"工具执行超过 {skill.timeout_seconds} 秒",
                True,
                "缩小输入范围后重试；若再次超时，切换降级工具或转人工复核。",
            )
            status = "failed"
            error_type = "ToolTimeoutError"
            self._record_failure(skill, circuit)
        except Exception as exc:
            result = self._error_payload(
                "TOOL_EXECUTION_ERROR",
                str(exc),
                True,
                "检查参数、依赖与权限；仅在输入不变且错误可重试时再次调用。",
            )
            status = "failed"
            error_type = exc.__class__.__name__
            self._record_failure(skill, circuit)
        finished = datetime.now().isoformat()
        duration_ms = round((time.perf_counter() - start_monotonic) * 1000, 2)
        input_size = len(json.dumps(payload, ensure_ascii=False, default=str))
        output_size = len(json.dumps(result, ensure_ascii=False, default=str))
        record = {
            "run_id": run_id,
            "request_id": current_request_id(),
            "tenant_id": principal.tenant_id,
            "subject": principal.subject,
            "skill": name,
            "status": status,
            "input": self._redact(payload),
            "output": self._redact(result),
            "started_at": started,
            "finished_at": finished,
            "duration_ms": duration_ms,
            "input_size": input_size,
            "output_size": output_size,
            "error_type": error_type,
            "estimated_cost": 0,
            "cache_hit": cache_hit,
            "circuit_state": circuit["state"],
            "validation_errors": validation_errors,
            "output_validation_errors": (
                output_validation_errors
                if "output_validation_errors" in locals()
                else []
            ),
            "authorization": {
                "required_permissions": skill.permissions,
                "missing_permissions": missing_permissions,
                "roles": list(principal.roles),
                "decision": "denied" if missing_permissions else "allowed",
            },
        }

        with self._log_lock:
            with self.log_file.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def recent_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        if not self.log_file.exists():
            return []
        records = [json.loads(line) for line in self.log_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        visible = [record for record in records if record_visible(record)]
        return visible[-limit:][::-1]

    def delete_run(self, run_id: str) -> bool:
        if not self.log_file.exists():
            return False
        lines = self.log_file.read_text(encoding="utf-8").splitlines()
        retained: List[str] = []
        removed = False
        for line in lines:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                retained.append(line)
                continue
            if record.get("run_id") == run_id and record_visible(record):
                removed = True
                continue
            retained.append(json.dumps(record, ensure_ascii=False))
        if removed:
            self.log_file.write_text("\n".join(retained) + ("\n" if retained else ""), encoding="utf-8")
        return removed

    def metrics(self, limit: int = 500) -> Dict[str, Any]:
        runs = self.recent_runs(limit)
        latencies = [float(run.get("duration_ms") or 0) for run in runs]
        successes = [run for run in runs if run.get("status") == "success"]
        failures_by_skill: Dict[str, int] = {}
        volume_by_skill: Dict[str, int] = {}
        for run in runs:
            skill_name = str(run.get("skill") or "unknown")
            volume_by_skill[skill_name] = volume_by_skill.get(skill_name, 0) + 1
            if run.get("status") != "success":
                failures_by_skill[skill_name] = failures_by_skill.get(skill_name, 0) + 1
        p95 = 0
        if len(latencies) >= 20:
            p95 = statistics.quantiles(latencies, n=20)[-1]
        elif latencies:
            p95 = max(latencies)
        circuits = {
            name: {
                "state": state.get("state", "closed"),
                "failures": state.get("failures", 0),
                "retry_after": max(0, round(float(state.get("open_until", 0)) - time.time(), 2)),
            }
            for name, state in self._circuits.items()
        }
        open_circuits = sum(1 for state in circuits.values() if state.get("state") == "open")
        return {
            "skills": len(self.skills),
            "total_runs": len(runs),
            "success_rate": round(len(successes) / max(len(runs), 1), 3),
            "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
            "p95_latency_ms": round(p95, 2),
            "failures_by_skill": failures_by_skill,
            "volume_by_skill": volume_by_skill,
            "estimated_cost": round(sum(float(run.get("estimated_cost") or 0) for run in runs), 4),
            "cache_hits": sum(1 for run in runs if run.get("cache_hit")),
            "open_circuits": open_circuits,
            "circuits": circuits,
        }

    def _register(self, skill: Skill) -> None:
        required = {
            "audit.scope_planner": ["scope", "objectives", "deliverables"],
            "audit.evidence_checklist": ["evidence_requests", "collection_methods"],
            "audit.finding_writer": ["condition", "criteria", "cause", "effect", "recommendation"],
            "rag.query": ["answer", "sources"],
            "audit.control_mapper": ["control_matrix", "test_procedures"],
            "agent.eval_case_designer": ["cases", "metrics"],
            "audit.remediation_planner": ["owner", "due_date", "acceptance_criteria"],
            "audit.sample_designer": ["sample_size", "selection_method", "expansion_rule"],
            "audit.exception_triage": ["severity", "root_cause", "escalation"],
            "audit.report_packager": ["delivery_manifest", "review_status"],
            "audit.deep_research_brief": ["query_rewrites", "source_strategy", "review_conditions"],
            "audit.delivery_verifier": ["verdict", "checks", "next_action"],
        }.get(skill.name, [])
        if required and not skill.output_schema:
            skill.output_schema = {
                "type": "object",
                "properties": {field: {} for field in required},
                "required": required,
            }
        self.skills[skill.name] = skill

    def _error_payload(
        self,
        code: str,
        message: str,
        retryable: bool,
        recovery: str,
        details: Any = None,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "error": message,
            "error_detail": {
                "code": code,
                "message": message,
                "retryable": retryable,
                "recovery": recovery,
                "details": details,
            },
        }

    _BEARER_PATTERN = re.compile(r"^Bearer\s+[A-Za-z0-9_\-]{12,}$")
    _OPENAI_KEY_PATTERN = re.compile(r"^sk-[A-Za-z0-9_\-]{16,}$")
    _AWS_ACCESS_KEY_PATTERN = re.compile(r"^AKIA[0-9A-Z]{16}$")

    def _looks_like_secret(self, value: str) -> bool:
        if not isinstance(value, str):
            return False
        if self._OPENAI_KEY_PATTERN.match(value):
            return True
        if self._AWS_ACCESS_KEY_PATTERN.match(value):
            return True
        if self._BEARER_PATTERN.match(value):
            return True
        return False

    def _redact(self, value: Any, key: str = "") -> Any:
        sensitive = {"api_key", "apikey", "token", "secret", "password", "authorization", "cookie"}
        if key.lower().replace("-", "_") in sensitive:
            return "[REDACTED]"
        if isinstance(value, dict):
            return {item_key: self._redact(item, str(item_key)) for item_key, item in value.items()}
        if isinstance(value, list):
            return [self._redact(item, key) for item in value]
        if isinstance(value, str) and self._looks_like_secret(value):
            return "[REDACTED]"
        return value

    def _describe(self, skill: Skill) -> Dict[str, Any]:
        return {
            "name": skill.name,
            "title": skill.title,
            "description": skill.description,
            "input_schema": skill.input_schema,
            "output_schema": skill.output_schema or {"type": "object"},
            "permissions": skill.permissions,
            "version": skill.version,
            "resilience": {
                "timeout_seconds": skill.timeout_seconds,
                "cache_ttl_seconds": skill.cache_ttl_seconds,
                "failure_threshold": skill.failure_threshold,
            },
        }

    def _cache_key(self, name: str, payload: Dict[str, Any]) -> str:
        return f"{current_principal().tenant_id}:{name}:{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"

    def _permission_allowed(self, required: str, permission_context: List[str] | None) -> bool:
        if permission_context is None:
            return has_permission(current_principal(), required)
        permissions = set(permission_context)
        if "*" in permissions or required in permissions:
            return True
        action, _, domain = required.partition(":")
        return f"{action}:*" in permissions or f"*:{domain}" in permissions

    def _cached(self, cache_key: str) -> bool:
        cached = self._cache.get(cache_key)
        if not cached:
            return False
        if cached[0] <= time.time():
            self._cache.pop(cache_key, None)
            return False
        return True

    def _record_failure(self, skill: Skill, circuit: Dict[str, Any]) -> None:
        circuit["failures"] = int(circuit.get("failures", 0)) + 1
        if circuit["failures"] >= skill.failure_threshold:
            circuit["state"] = "open"
            circuit["open_until"] = time.time() + 30

    def _validate(self, payload: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        errors = []
        for field in schema.get("required", []):
            if field not in payload or payload[field] in (None, "", [], {}):
                errors.append(f"missing required field: {field}")
        expected_types = {
            "string": str,
            "array": list,
            "object": dict,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
        }
        for field, definition in schema.get("properties", {}).items():
            if field not in payload:
                continue
            expected = expected_types.get(definition.get("type"))
            if expected and not isinstance(payload[field], expected):
                errors.append(f"{field} must be {definition.get('type')}")
        return errors

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
                    "properties": {"control_domain": {"type": "string"}, "risk_level": {"type": "string"}},
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
                    "properties": {"condition": {"type": "string"}, "criteria": {"type": "string"}, "risk": {"type": "string"}},
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
                timeout_seconds=12.0,
                cache_ttl_seconds=120,
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
                description="面向 Agent、RAG、工具调用和安全门禁场景生成可落地的评测用例。",
                input_schema={
                    "type": "object",
                    "properties": {"scenario": {"type": "string"}, "capabilities": {"type": "array", "items": {"type": "string"}}},
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
                    "properties": {"finding": {"type": "string"}, "severity": {"type": "string"}, "owner_role": {"type": "string"}},
                    "required": ["finding"],
                },
                permissions=["write:tasks"],
                handler=self._remediation_planner,
            )
        )

        self._register(
            Skill(
                name="audit.sample_designer",
                title="审计抽样方案生成",
                description="根据总体规模、风险等级、控制频率和证据类型生成可落地抽样方案。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "population": {"type": "integer"},
                        "risk_level": {"type": "string"},
                        "frequency": {"type": "string"},
                        "evidence_type": {"type": "string"},
                    },
                    "required": ["population"],
                },
                permissions=["read:evidence", "write:workpaper"],
                handler=self._sample_designer,
            )
        )
        self._register(
            Skill(
                name="audit.exception_triage",
                title="审计例外分级与处置",
                description="对控制测试例外进行严重性分级、根因归类、扩大样本和整改动作建议。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "finding": {"type": "string"},
                        "exceptions": {"type": "array", "items": {"type": "string"}},
                        "risk_level": {"type": "string"},
                    },
                    "required": ["finding"],
                },
                permissions=["write:workpaper", "write:tasks"],
                handler=self._exception_triage,
            )
        )
        self._register(
            Skill(
                name="audit.report_packager",
                title="审计报告交付打包",
                description="把审计结论、证据、控制测试、整改任务和复核意见整理为交付包目录。",
                input_schema={
                    "type": "object",
                    "properties": {"audit_item": {"type": "string"}, "run_id": {"type": "string"}},
                    "required": ["audit_item"],
                },
                permissions=["read:workpaper", "write:report"],
                handler=self._report_packager,
            )
        )
        self._register(
            Skill(
                name="audit.deep_research_brief",
                title="Deep Research 研究计划",
                description="为复杂审计问题生成查询改写、来源策略、证据问题和人工复核条件。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "standard": {"type": "string"},
                        "domain": {"type": "string"},
                    },
                    "required": ["question"],
                },
                permissions=["read:knowledge", "read:standards"],
                handler=self._deep_research_brief,
                cache_ttl_seconds=120,
            )
        )
        self._register(
            Skill(
                name="audit.delivery_verifier",
                title="审计交付验证",
                description="独立检查上游角色产物、证据链、控制覆盖和整改闭环，决定是否转人工复核。",
                input_schema={
                    "type": "object",
                    "properties": {
                        "audit_item": {"type": "string"},
                        "upstream_artifacts": {"type": "array", "items": {"type": "object"}},
                        "upstream_steps": {"type": "array", "items": {"type": "object"}},
                    },
                    "required": ["audit_item", "upstream_artifacts"],
                },
                permissions=["read:workpaper", "read:evidence"],
                handler=self._delivery_verifier,
            )
        )

    def _sample_designer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        population = max(1, int(payload.get("population") or 1))
        risk_level = str(payload.get("risk_level") or "medium").lower()
        frequency = payload.get("frequency") or "daily"
        evidence_type = payload.get("evidence_type") or "system log"
        base = 25 if risk_level in {"high", "critical", "高"} else 15 if risk_level in {"medium", "中"} else 8
        sample_size = min(population, max(base, round(population ** 0.5 * (2.2 if risk_level in {"high", "critical", "高"} else 1.5))))
        return {
            "population": population,
            "risk_level": risk_level,
            "frequency": frequency,
            "evidence_type": evidence_type,
            "sample_size": sample_size,
            "selection_method": "stratified_random_plus_key_item_review",
            "expansion_rule": "major exceptions expand the same-class sample; missing evidence pauses the conclusion",
            "method": "分层随机抽样 + 关键项全检" if risk_level in {"high", "critical", "高"} else "随机抽样 + 异常定向补样",
            "strata": ["高权限/高金额/高影响记录", "普通运行记录", "期间首末与变更窗口记录"],
            "exception_handling": "发现重大例外时扩大样本并触发复核；证据缺失时进入人工补证和整改任务。",
        }

    def _exception_triage(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        finding = payload.get("finding", "")
        exceptions = payload.get("exceptions") or []
        risk_level = str(payload.get("risk_level") or "medium").lower()
        severity = "high" if risk_level in {"high", "critical", "高"} or len(exceptions) >= 3 else "medium" if exceptions else "low"
        return {
            "finding": finding,
            "severity": severity,
            "exception_count": len(exceptions),
            "root_cause": "segregation of duties, approval flow, or log review is incomplete",
            "escalation": "high-risk or three-plus exceptions require manager review and sample expansion",
            "root_causes": ["职责分离不足", "审批链路不完整", "日志留存或复核机制薄弱"],
            "next_actions": [
                "补充关键证据并标记无法追溯样本",
                "扩大样本覆盖同类交易或权限变更",
                "生成整改任务并设置关闭验收标准",
                "重大例外提交审计经理复核",
            ],
        }

    def _report_packager(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        audit_item = payload.get("audit_item") or "审计项目"
        run_id = payload.get("run_id") or "draft"
        return {
            "audit_item": audit_item,
            "run_id": run_id,
            "delivery_manifest": [
                {"order": index + 1, "section": section, "status": "ready"}
                for index, section in enumerate(
                    ["scope", "control mapping", "evidence and sampling", "findings", "remediation", "appendices"]
                )
            ],
            "review_status": "pending_independent_review",
            "sections": [
                "01 项目背景与范围",
                "02 控制矩阵与标准映射",
                "03 证据清单与抽样底稿",
                "04 风险结论与审计发现",
                "05 整改计划与复核关闭",
                "06 交付包索引与附录",
            ],
            "quality_checks": ["来源可追溯", "表格字段齐全", "风险评分一致", "整改责任明确", "下载文件可复核"],
            "download_artifacts": [f"{run_id}-audit-report.md", f"{run_id}-delivery-pack.md"],
        }

    def _deep_research_brief(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        question = payload.get("question", "")
        standard = payload.get("standard") or "ISO27001/SOX/COBIT"
        domain = payload.get("domain") or "enterprise audit"
        return {
            "question": question,
            "standard": standard,
            "domain": domain,
            "review_conditions": [
                "conflicting sources, unknown standard versions, or fewer than two independent sources",
                "production access, personal data, or material financial impact",
            ],
            "query_rewrites": [
                f"{question} {standard} audit evidence",
                f"{domain} control testing checklist remediation",
                f"{standard} risk assessment audit workpaper",
            ],
            "source_strategy": ["内部知识库优先", "标准条款与控制库交叉验证", "历史审计案例补充", "低置信度结论触发人工复核"],
            "evidence_questions": ["需要哪些设计证据？", "需要哪些运行证据？", "哪些证据缺失会影响结论？", "哪些发现需要整改闭环？"],
        }

    def _delivery_verifier(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        artifacts = [item for item in payload.get("upstream_artifacts", []) if isinstance(item, dict)]
        steps = [item for item in payload.get("upstream_steps", []) if isinstance(item, dict)]
        artifact_types = {str(item.get("type") or "") for item in artifacts}
        required_types = {"audit", "evidence", "risk"}
        missing_types = sorted(required_types - artifact_types)
        failed_steps = [item.get("step_id") for item in steps if item.get("status") != "success"]
        integrity_failures = []
        for artifact in artifacts:
            content = artifact.get("content")
            canonical = json.dumps(content, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
            actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
            if actual != artifact.get("content_hash"):
                integrity_failures.append(artifact.get("artifact_id"))
        contents = [item.get("content") for item in artifacts if isinstance(item.get("content"), dict)]
        control_rows = [
            row
            for content in contents
            for row in content.get("control_matrix", [])
            if isinstance(row, dict)
        ]
        evidence_requests = [
            row
            for content in contents
            for row in content.get("evidence_requests", [])
            if isinstance(row, dict)
        ]
        remediation_outputs = [
            content
            for content in contents
            if all(content.get(field) not in (None, "", [], {}) for field in ("owner", "due_date", "acceptance_criteria"))
        ]
        checks = [
            {"check": "upstream_artifacts", "passed": len(artifacts) >= 4, "evidence": len(artifacts)},
            {"check": "required_domains", "passed": not missing_types, "evidence": sorted(artifact_types)},
            {"check": "step_failures", "passed": not failed_steps, "evidence": failed_steps},
            {
                "check": "provenance",
                "passed": all(item.get("source_run_id") and item.get("producer") for item in artifacts),
                "evidence": [item.get("producer") for item in artifacts],
            },
            {"check": "artifact_integrity", "passed": not integrity_failures, "evidence": integrity_failures},
            {"check": "control_coverage", "passed": bool(control_rows), "evidence": len(control_rows)},
            {"check": "evidence_design", "passed": bool(evidence_requests), "evidence": len(evidence_requests)},
            {"check": "remediation_closure", "passed": bool(remediation_outputs), "evidence": len(remediation_outputs)},
        ]
        passed = sum(1 for item in checks if item["passed"])
        raw_score = passed / max(len(checks), 1)
        return {
            "audit_item": payload.get("audit_item"),
            "verdict": "pass" if raw_score >= 0.8 else "human_review",
            "raw_score": round(raw_score, 3),
            "checks": checks,
            "missing_artifact_types": missing_types,
            "failed_steps": failed_steps,
            "next_action": "进入报告交付" if raw_score >= 0.8 else "补齐缺失产物并由审计经理复核",
        }

    def _scope_planner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = payload.get("audit_item", "待审计对象")
        standard = payload.get("standard") or payload.get("standard_type") or "ISO27001/COBIT"
        topics = payload.get("risk_topics") or ["权限", "变更", "日志", "数据"]
        return {
            "objectives": [
                "验证控制设计是否覆盖关键风险",
                "验证运行证据是否充分且可追溯",
                "形成可分派、可验收的整改闭环",
            ],
            "scope": f"{item} 的关键流程、权限、变更、日志、接口和数据处理活动",
            "standard": standard,
            "risk_topics": topics,
            "deliverables": ["审计范围说明", "控制矩阵", "抽样计划", "发现草稿", "整改任务"],
            "human_review": "范围涉及高风险系统或生产数据时，需要审计经理复核后进入现场阶段。",
        }

    def _evidence_checklist(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        domain = payload.get("control_domain", "访问控制")
        common = ["制度文件", "审批记录", "系统配置截图", "日志样本", "抽样底稿", "复核记录"]
        if "变更" in domain:
            common.extend(["变更单", "测试报告", "回退方案"])
        if "数据" in domain:
            common.extend(["数据目录", "分类分级规则", "加密/脱敏配置"])
        if "权限" in domain or "访问" in domain:
            common.extend(["用户清单", "角色矩阵", "离职禁用记录", "特权账号复核记录"])
        return {
            "control_domain": domain,
            "evidence": common,
            "collection_method": "系统导出 + 访谈 + 抽样核验",
            "evidence_requests": [
                {"evidence": item, "source": domain, "purpose": "control_design_or_operation_test"}
                for item in common
            ],
            "collection_methods": ["system_export", "interview", "sample_reperformance"],
            "quality_rule": "每项关键控制至少需要一项设计证据和一项运行证据，缺失时触发人工复核。",
        }

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
            "recommendation": "补齐控制证据，并建立周期性复核和例外跟踪机制。",
        }

    def _rag_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from rag.agentic_rag import RAGPipeline

        return RAGPipeline().query(payload["question"])

    def _control_mapper(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        item = payload.get("audit_item", "待审计对象")
        standard = payload.get("standard") or payload.get("standard_type") or "ISO27001"
        topics = payload.get("risk_topics") or ["权限", "变更", "日志"]
        controls = []
        for index, topic in enumerate(topics, start=1):
            controls.append(
                {
                    "control_id": f"MAP-{index:02d}",
                    "domain": topic,
                    "objective": f"确认 {item} 在 {topic} 领域满足 {standard} 相关控制要求",
                    "test_procedure": "检查制度设计、抽样验证执行记录、复核例外审批，并追踪整改闭环。",
                    "evidence_required": ["制度或流程文件", "审批记录", "系统配置截图", "抽样底稿", "复核记录"],
                    "quality_rule": "每项控制至少需要一项设计证据和一项运行证据，否则进入人工复核。",
                }
            )
        return {
            "audit_item": item,
            "standard": standard,
            "control_matrix": controls,
            "test_procedures": [control["test_procedure"] for control in controls],
        }

    def _eval_case_designer(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        scenario = payload.get("scenario", "企业审计 Agent")
        capabilities = payload.get("capabilities") or ["RAG", "工具调用", "质量门", "人工复核"]
        cases = []
        for index, capability in enumerate(capabilities, start=1):
            cases.append(
                {
                    "case_id": f"EVAL-{index:02d}",
                    "capability": capability,
                    "question": f"在 {scenario} 中验证 {capability} 能力是否可用",
                    "expected_terms": ["证据", "来源", "风险", "结论"],
                    "pass_rule": "回答必须引用来源、给出风险判断，并说明缺失证据或人工复核条件。",
                }
            )
        return {"scenario": scenario, "cases": cases, "metrics": ["retrieval_relevance", "faithfulness", "tool_success", "human_review_trigger"]}

    def _remediation_planner(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        severity = payload.get("severity", "中")
        due_days = 7 if severity in {"高", "high", "critical"} else 14 if severity in {"中", "medium"} else 30
        return {
            "title": payload.get("finding", "审计发现整改"),
            "owner_role": payload.get("owner_role", "控制责任人"),
            "owner": payload.get("owner_role", "控制责任人"),
            "due_days": due_days,
            "due_date": (datetime.now() + timedelta(days=due_days)).date().isoformat(),
            "tasks": ["确认影响范围和责任人", "补齐控制设计和运行证据", "完成例外审批或权限清理", "由审计或内控团队复核关闭"],
            "acceptance_criteria": "整改证据完整、抽样无重大例外、复核意见已记录。",
            "status_flow": ["未开始", "进行中", "待验证", "已完成", "已关闭"],
        }

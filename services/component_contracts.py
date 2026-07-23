"""Executable component boundaries for the AuditPilot agent platform.

The catalog is intentionally business-facing.  It documents ownership,
inputs, outputs, invariants, and evaluation responsibilities without tying
the product to a specific orchestration framework.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List


_COMPONENTS: List[Dict[str, Any]] = [
    {
        "id": "task_specification",
        "name": "任务规格",
        "owner": "Agent Runtime",
        "purpose": "把审计目标转换为可执行、可停止、可复核的任务契约。",
        "inputs": ["objective", "audit_context", "budgets"],
        "outputs": ["task_envelope", "success_criteria", "stop_conditions"],
        "invariants": ["目标不能为空", "工具与重试预算必须有上限", "任务必须具备人工复核出口"],
        "does_not_own": ["模型回答生成", "证据内容判定", "发布审批"],
        "evaluators": ["objective_clarity", "budget_bounded", "review_exit"],
        "weight": 0.10,
        "critical": True,
    },
    {
        "id": "agent_loop",
        "name": "任务编排",
        "owner": "Bounded Agent Loop",
        "purpose": "按依赖执行计划步骤，并在完成、阻断、需复核或预算耗尽时停止。",
        "inputs": ["task_envelope", "plan", "step_results"],
        "outputs": ["trajectory", "termination_reason", "artifacts"],
        "invariants": ["不得绕过依赖", "不得无限循环", "失败后不得静默继续"],
        "does_not_own": ["工具内部实现", "知识库索引", "人工审批结论"],
        "evaluators": ["dependency_conformance", "termination_safety", "step_completion"],
        "weight": 0.14,
        "critical": True,
    },
    {
        "id": "tool_runtime",
        "name": "工具执行",
        "owner": "Skill Registry",
        "purpose": "校验输入、执行受权限约束的工具，并记录延迟、重试、缓存和熔断。",
        "inputs": ["tool_name", "validated_arguments", "permission_context"],
        "outputs": ["tool_result", "execution_span", "recovery_signal"],
        "invariants": ["输入必须通过 Schema 校验", "敏感字段必须脱敏", "失败必须结构化返回"],
        "does_not_own": ["任务拆解", "业务结论", "发布门禁"],
        "evaluators": ["schema_validity", "tool_success", "retry_discipline", "latency"],
        "weight": 0.13,
        "critical": True,
    },
    {
        "id": "evidence_grounding",
        "name": "检索与证据",
        "owner": "RAG / Evidence Services",
        "purpose": "召回可引用的制度、底稿与系统证据，并保留来源和证据缺口。",
        "inputs": ["audit_question", "scope", "source_policy"],
        "outputs": ["ranked_sources", "citations", "evidence_gaps"],
        "invariants": ["来源必须可追溯", "低置信度不得输出绝对结论", "缺失证据必须显式呈现"],
        "does_not_own": ["整改责任分配", "工具授权", "模型版本发布"],
        "evaluators": ["source_coverage", "authority", "faithfulness", "gap_visibility"],
        "weight": 0.14,
        "critical": True,
    },
    {
        "id": "evidence_graph",
        "name": "证据关系图",
        "owner": "Evidence Graph",
        "purpose": "连接任务、计划、工具、产物与复核，形成可验证的审计血缘。",
        "inputs": ["episode_package", "control_dependencies", "artifact_refs"],
        "outputs": ["nodes", "edges", "lineage_metrics"],
        "invariants": ["产物必须有来源", "依赖边必须指向存在节点", "关键步骤不得成为孤点"],
        "does_not_own": ["通用向量召回", "图数据库运维", "最终审计意见"],
        "evaluators": ["provenance_coverage", "broken_dependency_rate", "orphan_rate"],
        "weight": 0.10,
        "critical": False,
    },
    {
        "id": "safety_governance",
        "name": "安全与权限",
        "owner": "Safety Gate",
        "purpose": "在任务和步骤边界执行输入检查、权限约束和人工升级。",
        "inputs": ["task_payload", "tool_permissions", "stage"],
        "outputs": ["gate_decision", "findings", "required_intervention"],
        "invariants": ["阻断项不得执行", "高风险动作必须留下复核记录", "凭据不得进入轨迹"],
        "does_not_own": ["业务风险评分", "模型推理", "知识内容维护"],
        "evaluators": ["blocked_action_prevention", "review_trigger", "secret_redaction"],
        "weight": 0.13,
        "critical": True,
    },
    {
        "id": "memory_context",
        "name": "上下文与记忆",
        "owner": "Context / Memory",
        "purpose": "管理任务上下文、检查点和经批准的经验，避免无边界记忆污染。",
        "inputs": ["working_context", "task_events", "approved_lessons"],
        "outputs": ["bounded_context", "checkpoint", "relevant_lessons"],
        "invariants": ["原始敏感上下文不得进入评测包", "经验必须有来源", "未批准经验不得自动应用"],
        "does_not_own": ["工具执行", "知识源权威性", "审批决策"],
        "evaluators": ["context_minimization", "checkpoint_presence", "lesson_governance"],
        "weight": 0.08,
        "critical": False,
    },
    {
        "id": "audit_delivery",
        "name": "审计交付",
        "owner": "Audit Delivery",
        "purpose": "将控制、证据、发现和整改转换为可下载、可复核的交付包。",
        "inputs": ["validated_artifacts", "findings", "review_decision"],
        "outputs": ["workpaper_index", "report", "remediation_tracker"],
        "invariants": ["结论必须关联证据", "发现必须包含整改动作", "未复核结论不得标记为最终"],
        "does_not_own": ["模型调用", "工具注册", "经验推广"],
        "evaluators": ["artifact_coverage", "remediation_actionability", "review_state"],
        "weight": 0.10,
        "critical": False,
    },
    {
        "id": "improvement_governance",
        "name": "持续改进",
        "owner": "Evaluation Harness",
        "purpose": "把失败和反思沉淀为候选经验，经回归评测和人工批准后再复用。",
        "inputs": ["traces", "component_scores", "human_feedback"],
        "outputs": ["experience_candidate", "regression_result", "promotion_decision"],
        "invariants": ["评测器不可被候选修改", "必须同时通过基线与留出集", "推广必须人工审批"],
        "does_not_own": ["在线任务执行", "原始证据存储", "模型训练平台"],
        "evaluators": ["failure_attribution", "heldout_gate", "human_approval"],
        "weight": 0.08,
        "critical": False,
    },
]


def component_catalog() -> List[Dict[str, Any]]:
    """Return an isolated copy so callers cannot mutate platform contracts."""

    return deepcopy(_COMPONENTS)


def component_by_id(component_id: str) -> Dict[str, Any]:
    for component in _COMPONENTS:
        if component["id"] == component_id:
            return deepcopy(component)
    raise KeyError(component_id)

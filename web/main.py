"""FastAPI web application for the Intelligent Audit System."""

from __future__ import annotations

import json
import inspect
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from agents.audit_agent import AuditAgent, CONTROL_LIBRARY
from config import LLM_CONFIG, PATHS, SECURITY_CONFIG, UPLOAD_CONFIG, WEB_CONFIG
from knowledge_graph.builder import KnowledgeGraphBuilder
from services.agent_runtime import AgentRuntime
from services.audit_delivery import AuditDeliveryService
from services.audit_repository import AuditRunRepository
from services.audit_templates import list_audit_templates
from services.conversation_memory import ConversationMemory
from services.evaluation_orchestrator import ComponentEvaluationOrchestrator
from services.evaluation_repository import EvaluationRunRepository
from services.evidence_analyzer import EvidenceAnalyzer
from services.experience_curator import ExperienceCurator
from services.evolution_harness import EvolutionHarness
from services.harness_control import HarnessControlPlane
from services.http_middleware import SecurityAuditMiddleware
from services.agent_quality import AgentQualityDiagnostics
from services.intent_router import HybridIntentRouter
from services.product_insights import ProductInsights
from services.rag_evaluator import RAGEvaluator
from services.research_agent import AuditResearchAgent
from services.safety_gate import SafetyGate
from services.skill_registry import SkillRegistry
from services.security import AuditEventStore, current_principal, current_tenant_id
from services.upload_security import read_validated_upload


logger = logging.getLogger(__name__)

audit_agent: Optional[AuditAgent] = None
rag_pipeline = None
kg_builder: Optional[KnowledgeGraphBuilder] = None
evaluator = None
audit_repository = AuditRunRepository()
skill_registry = SkillRegistry()
safety_gate = SafetyGate()
agent_runtime = AgentRuntime(skill_registry, safety_gate)
product_insights = ProductInsights(audit_repository, skill_registry)
audit_delivery = AuditDeliveryService(audit_repository)
evaluation_repository = EvaluationRunRepository()
experience_curator = ExperienceCurator()
evidence_analyzer = EvidenceAnalyzer()
conversation_memory = ConversationMemory()
intent_router = HybridIntentRouter()
harness_control = HarnessControlPlane()
evolution_harness = EvolutionHarness(evaluation_repository, agent_runtime, skill_registry, conversation_memory, harness_control)
agent_quality = AgentQualityDiagnostics(
    evaluation_repository,
    agent_runtime,
    skill_registry,
    conversation_memory,
    harness_control,
)
component_evaluator = ComponentEvaluationOrchestrator(
    agent_runtime,
    evaluation_repository,
    skill_registry,
)
evaluation_cache: Dict[str, Any] = {}
research_cache: Dict[str, Any] = {}


def cache_key(prefix: str, payload: Any) -> str:
    return f"{prefix}:{json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)}"


def clone_payload(payload: Any) -> Any:
    return json.loads(json.dumps(payload, ensure_ascii=False, default=str))


def _search_text(*values: Any) -> str:
    return " ".join(str(value or "") for value in values).lower()


def _search_item(item_type: str, title: str, subtitle: str, href: str, keywords: str = "", created_at: str = "", badge: str = "") -> Dict[str, Any]:
    return {
        "type": item_type,
        "title": title,
        "subtitle": subtitle,
        "href": href,
        "keywords": keywords,
        "created_at": created_at,
        "badge": badge,
    }


def _search_timestamp(value: str) -> float:
    if not value:
        return 0
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return 0


def collect_search_results(query: str = "", limit: int = 12) -> List[Dict[str, Any]]:
    normalized = str(query or "").strip().lower()
    max_items = max(1, min(int(limit or 12), 50))
    static_items = [
        _search_item("command", "新建审计项目", "进入审计项目工作台，生成范围、取证、控制测试和整改闭环", "/audit", "audit project create 审计 立项"),
        _search_item("command", "Agent 协作", "多角色 Agent 路由、证据、控制、风险和整改协作", "/chat", "chat multi agent route 协作"),
        _search_item("command", "知识与证据", "管理 RAG 知识、证据文件分析和知识图谱", "/knowledge", "rag evidence knowledge graph 证据 知识"),
        _search_item("command", "Agent 运行时", "查看任务、工具、MCP、Harness、自进化和运行日志", "/skills", "runtime tool use mcp harness 自进化"),
        _search_item("command", "评测与发布门禁", "运行 Agent/RAG/Deep Research 评测并查看发布门禁", "/training", "evaluation release gate rag 评测"),
    ]

    dynamic_items: List[Dict[str, Any]] = []
    for run in audit_repository.list_runs(limit=80):
        dynamic_items.append(
            _search_item(
                "audit",
                str(run.get("audit_item") or "审计项目"),
                f"{run.get('run_id')} · {run.get('lifecycle_stage') or '-'} · 风险 {run.get('risk_level') or '-'} · 合规 {run.get('compliance_score') or '-'}",
                f"/audit?run_id={run.get('run_id')}",
                _search_text(run.get("run_id"), run.get("audit_item"), run.get("audit_type"), run.get("status"), run.get("risk_level")),
                str(run.get("created_at") or ""),
                str(run.get("status") or ""),
            )
        )
    for run in evaluation_repository.list_runs(limit=80):
        metrics = run.get("metrics", {})
        gate = run.get("release_gate", {})
        dynamic_items.append(
            _search_item(
                "evaluation",
                f"{run.get('run_type')} 评测记录",
                f"{run.get('run_id')} · 得分 {metrics.get('overall_score', '-')} · 门禁 {gate.get('label') or gate.get('status') or '-'}",
                f"/training?run_id={run.get('run_id')}",
                _search_text(run.get("run_id"), run.get("run_type"), gate.get("label"), gate.get("status")),
                str(run.get("created_at") or ""),
                str(gate.get("label") or gate.get("status") or ""),
            )
        )
    for analysis in evidence_analyzer.list_analyses(limit=80):
        gate = analysis.get("quality_gate", {})
        dynamic_items.append(
            _search_item(
                "evidence",
                str(analysis.get("file_name") or "证据分析"),
                f"{analysis.get('analysis_id')} · 风险 {analysis.get('risk_count', 0)} · 控制 {analysis.get('control_count', 0)} · {gate.get('label') or gate.get('status') or '-'}",
                f"/audit?analysis_id={analysis.get('analysis_id')}",
                _search_text(analysis.get("analysis_id"), analysis.get("file_name"), gate.get("label"), gate.get("status")),
                str(analysis.get("created_at") or ""),
                str(gate.get("label") or gate.get("status") or ""),
            )
        )
    for task in agent_runtime.list_tasks(limit=80):
        dynamic_items.append(
            _search_item(
                "task",
                str(task.get("objective") or task.get("task_id")),
                f"{task.get('task_id')} · {task.get('status')} · 步骤 {len(task.get('steps', []))}/{len(task.get('plan', []))}",
                f"/skills?task_id={task.get('task_id')}",
                _search_text(task.get("task_id"), task.get("objective"), task.get("status"), task.get("context")),
                str(task.get("created_at") or ""),
                str(task.get("status") or ""),
            )
        )
    for run in skill_registry.recent_runs(limit=80):
        dynamic_items.append(
            _search_item(
                "tool",
                str(run.get("skill") or "Skill 运行"),
                f"{run.get('run_id')} · {run.get('status')} · {run.get('duration_ms', 0)}ms",
                "/skills",
                _search_text(run.get("run_id"), run.get("skill"), run.get("status"), run.get("error_type")),
                str(run.get("started_at") or ""),
                str(run.get("status") or ""),
            )
        )

    candidates = static_items + dynamic_items
    if normalized:
        candidates = [item for item in candidates if normalized in _search_text(item.get("title"), item.get("subtitle"), item.get("keywords"), item.get("badge"))]
    candidates.sort(key=lambda item: (0 if item["type"] == "command" else 1, -_search_timestamp(item.get("created_at", ""))))
    return candidates[:max_items]


def prewarm_evaluation_runtime() -> None:
    global audit_agent
    try:
        benchmark = get_evaluator()
        benchmark.evaluate_agent(
            [
                {
                    "id": "PREWARM-AGENT",
                    "category": "runtime_prewarm",
                    "question": "ERP 权限审计需要哪些证据、控制测试和整改动作？",
                    "expected_answer": "证据 控制 整改 复核",
                    "expected_terms": ["证据", "控制", "整改", "复核"],
                }
            ]
        )
        rag = init_rag_lazy()
        RAGEvaluator(rag).evaluate(
            [
                {
                    "case_id": "PREWARM-RAG",
                    "question": "ERP 权限审计需要哪些访问控制证据？",
                    "expected_terms": ["权限", "审批", "日志", "复核"],
                    "category": "runtime_prewarm",
                }
            ]
        )
        if audit_agent is None:
            audit_agent = AuditAgent(rag_pipeline=rag, enable_llm=False)
        audit_agent.process_audit_query(
            "ERP 权限审计需要哪些证据、控制测试和整改动作？",
            session_id="runtime_prewarm",
            prefer_llm=False,
        )
        logger.info("Evaluation runtime prewarmed")
    except Exception as exc:
        logger.info("Evaluation prewarm skipped: %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
    threading.Thread(target=prewarm_evaluation_runtime, daemon=True).start()
    logger.info("Intelligent Audit System started")
    yield
    global audit_agent, kg_builder
    if audit_agent:
        audit_agent.close()
    if kg_builder:
        kg_builder.close()
    logger.info("Intelligent Audit System stopped")


app = FastAPI(
    title="审脉 AuditPilot",
    description="面向审计交付场景的 Agentic RAG、风险评估、控制测试和整改闭环系统",
    version="4.0.0",
    lifespan=lifespan,
)

audit_event_store = AuditEventStore()
app.add_middleware(SecurityAuditMiddleware, audit_events=audit_event_store)
app.add_middleware(
    CORSMiddleware,
    allow_origins=WEB_CONFIG["cors_origins"] or ["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(PATHS["static"])), name="static")
templates = Jinja2Templates(directory=str(PATHS["templates"]))


def render_page(request: Request, name: str) -> Response:
    """Render across the supported Starlette template API transition."""

    parameters = inspect.signature(templates.TemplateResponse).parameters
    if next(iter(parameters), "") == "request":
        return templates.TemplateResponse(request=request, name=name)
    return templates.TemplateResponse(name, {"request": request})


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/security/session")
async def security_session_api():
    principal = current_principal()
    return {
        "success": True,
        "mode": SECURITY_CONFIG["mode"],
        "tenant_isolation": bool(SECURITY_CONFIG["tenant_isolation"]),
        "principal": principal.public_dict(),
        "timestamp": datetime.now().isoformat(),
    }


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None
    llm_enhance: bool = False


class RoutePreviewRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)


class AuditRequest(BaseModel):
    audit_item: str = Field(..., min_length=1, max_length=500)
    audit_type: str = Field(..., min_length=1, max_length=200)
    standard_type: Optional[str] = None
    risk_level: Optional[str] = None
    business_context: Optional[str] = Field(None, max_length=4000)
    audit_scope: Optional[str] = Field(None, max_length=4000)
    audit_period: Optional[str] = Field(None, max_length=500)
    key_questions: Optional[str] = Field(None, max_length=4000)
    existing_evidence: Optional[str] = Field(None, max_length=4000)


class KnowledgeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200000)
    metadata: Optional[Dict[str, Any]] = None


class EvaluationRequest(BaseModel):
    model_path: str = "current-agent"
    test_cases: Optional[List[Dict[str, Any]]] = None


class RAGEvaluationRequest(BaseModel):
    cases: Optional[List[Dict[str, Any]]] = None


class ResearchRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    context: Optional[Dict[str, Any]] = None
    persist_evaluation: bool = False


class SkillRunRequest(BaseModel):
    input: Dict[str, Any] = Field(default_factory=dict)


class AgentTaskRequest(BaseModel):
    objective: str = Field(..., min_length=1, max_length=2000)
    context: Dict[str, Any] = Field(default_factory=dict)


class AgentTaskStepRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    stage: str = Field("manual", max_length=100)
    skill: str = Field(..., min_length=1, max_length=200)
    purpose: str = Field("", max_length=1000)


class AgentLoopRequest(BaseModel):
    max_steps: int = Field(default=6, ge=1, le=12)


class ExperienceReviewRequest(BaseModel):
    decision: str = Field(..., pattern="^(approved|rejected)$")
    reviewer: str = Field("human-reviewer", max_length=200)
    comment: str = Field("", max_length=4000)


class SafetyGateRequest(BaseModel):
    payload: Dict[str, Any] = Field(default_factory=dict)
    stage: str = Field("runtime", max_length=100)


class HarnessEvaluationRequest(BaseModel):
    baseline: Dict[str, float] = Field(default_factory=dict)
    held_in: Dict[str, float] = Field(default_factory=dict)
    held_out: Dict[str, float] = Field(default_factory=dict)
    checks: List[Dict[str, Any]] = Field(default_factory=list)


class HarnessRunEvaluationRequest(BaseModel):
    baseline_run_id: str = Field(..., min_length=3, max_length=120)
    held_in_run_id: str = Field(..., min_length=3, max_length=120)
    held_out_run_id: str = Field(..., min_length=3, max_length=120)
    checks: List[Dict[str, Any]] = Field(default_factory=list)


class HarnessReviewRequest(BaseModel):
    decision: str = Field(..., pattern="^(approve|reject)$")
    reviewer: str = Field("Human Reviewer", max_length=200)
    comment: str = Field("", max_length=4000)


class ReviewRequest(BaseModel):
    reviewer: str = "复核人"
    decision: str = Field(..., pattern="^(approve|reject|need_evidence)$")
    comment: str = Field("", max_length=4000)


class TaskUpdateRequest(BaseModel):
    status: str = Field(..., max_length=100)
    owner: str = ""
    note: str = Field("", max_length=2000)


class EvidenceUpdateRequest(BaseModel):
    status: str = Field(..., max_length=100)
    owner: str = ""
    note: str = Field("", max_length=2000)


class ControlTestUpdateRequest(BaseModel):
    result: str = Field(..., max_length=100)
    tester: str = ""
    exception: str = Field("", max_length=2000)


class EvidenceAttachRequest(BaseModel):
    analysis_id: str = Field(..., min_length=1, max_length=100)


def init_rag_lazy():
    global rag_pipeline
    if rag_pipeline is None:
        from rag.agentic_rag import RAGPipeline

        rag_pipeline = RAGPipeline()
    return rag_pipeline


def get_audit_agent() -> AuditAgent:
    global audit_agent
    if audit_agent is None:
        audit_agent = AuditAgent(rag_pipeline=init_rag_lazy(), enable_llm=False)
    return audit_agent


def get_rag_pipeline():
    return init_rag_lazy()


def get_research_agent() -> AuditResearchAgent:
    return AuditResearchAgent(init_rag_lazy())


def get_kg_builder() -> KnowledgeGraphBuilder:
    global kg_builder
    if kg_builder is None:
        kg_builder = KnowledgeGraphBuilder()
    return kg_builder


def get_evaluator():
    global evaluator
    if evaluator is None:
        from training.training_pipeline import BenchmarkEvaluator

        evaluator = BenchmarkEvaluator()
    return evaluator


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return render_page(request, "index.html")


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return render_page(request, "chat.html")


@app.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    return render_page(request, "audit.html")


@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(request: Request):
    return render_page(request, "knowledge.html")


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    return render_page(request, "training.html")


@app.get("/skills", response_class=HTMLResponse)
async def skills_page(request: Request):
    return render_page(request, "skills.html")


@app.post("/api/chat")
async def chat_api(request: ChatRequest, agent: AuditAgent = Depends(get_audit_agent)):
    session_id = request.session_id or str(uuid.uuid4())
    routing = intent_router.classify(request.message)
    memory_context = conversation_memory.context_for(session_id, request.message)
    if request.context:
        memory_context["request_context"] = request.context
        memory_context["prompt_text"] = (
            f"{memory_context.get('prompt_text', '')}\n\n[请求上下文]\n"
            f"{json.dumps(request.context, ensure_ascii=False)}"
        ).strip()
    result = agent.process_audit_query(
        request.message,
        session_id=session_id,
        external_context=memory_context,
        prefer_llm=request.llm_enhance,
    )
    memory = conversation_memory.record_turn(
        session_id,
        request.message,
        result.get("response", ""),
        {"intent": routing["intent"], "agents": routing["agents"]},
    )
    return {
        "success": True,
        "session_id": session_id,
        "routing": routing,
        "memory": memory,
        "timestamp": datetime.now().isoformat(),
        **result,
    }


@app.post("/api/audit")
async def audit_api(request: AuditRequest, agent: AuditAgent = Depends(get_audit_agent)):
    audit_query = f"请对 {request.audit_item} 进行 {request.audit_type}"
    if request.standard_type:
        audit_query += f"，参考 {request.standard_type} 标准"
    if request.risk_level:
        audit_query += f"，关注 {request.risk_level} 风险"
    if request.audit_period:
        audit_query += f"。审计期间：{request.audit_period}"
    if request.business_context:
        audit_query += f"。业务背景：{request.business_context}"
    if request.audit_scope:
        audit_query += f"。审计范围：{request.audit_scope}"
    if request.key_questions:
        audit_query += f"。重点问题：{request.key_questions}"
    if request.existing_evidence:
        audit_query += f"。已有证据：{request.existing_evidence}"
    result = agent.process_audit_query(audit_query)
    if str(request.risk_level).lower() in {"高", "high", "critical"}:
        result["risk_assessment"]["risk_level"] = "高"
        result["risk_assessment"]["risk_score"] = max(float(result["risk_assessment"].get("risk_score") or 0), 0.72)
        result["quality_gate"]["escalation_required"] = True
    run = audit_repository.create_run(request.model_dump(), result)
    return {
        "success": True,
        "run_id": run["run_id"],
        "audit_item": request.audit_item,
        "audit_type": request.audit_type,
        "result": result,
        "run": run,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/audit/controls")
async def audit_controls_api():
    return {"success": True, "controls": CONTROL_LIBRARY, "timestamp": datetime.now().isoformat()}


@app.get("/api/audit/templates")
async def audit_templates_api():
    return {"success": True, "templates": list_audit_templates(), "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/capabilities")
async def agent_capabilities_api():
    return {
        "success": True,
        "capabilities": {
            "agent_architecture": [
                "任务规划",
                "Skill 注册与执行",
                "MCP 风格工具描述",
                "A2A 风格任务协议",
                "Agent Runtime",
                "工具调用",
                "Agentic RAG",
                "Self-Evolution Harness",
                "安全门禁",
                "质量门",
                "人工复核闭环",
            ],
            "rag": ["混合检索", "查询扩展", "来源引用", "降级检索", "RAG 评测"],
            "engineering": ["FastAPI", "持久化审计档案", "报告导出", "健康检查", "Docker 部署", "运行时观测", "工具成功率与延迟指标"],
            "audit_business": ["审计程序", "抽样计划", "证据请求中心", "控制测试工作台", "审计发现草稿", "整改任务跟踪"],
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/research/answer")
async def research_answer_api(request: ResearchRequest, research: AuditResearchAgent = Depends(get_research_agent)):
    key = cache_key("research", {"question": request.question, "context": request.context})
    if key in research_cache:
        result = clone_payload(research_cache[key])
        result["cache_hit"] = True
    else:
        result = research.answer(request.question, request.context)
        research_cache[key] = clone_payload(result)
        result["cache_hit"] = False
    run = None
    if request.persist_evaluation:
        run = evaluation_repository.create_run("research", request.model_dump(), result)
    return {"success": True, "result": result, "run": run, "timestamp": datetime.now().isoformat()}


@app.get("/api/research/jd-coverage")
async def research_jd_coverage_api(research: AuditResearchAgent = Depends(get_research_agent)):
    return {"success": True, "coverage": research.jd_coverage(), "timestamp": datetime.now().isoformat()}


@app.get("/api/research/evaluation-plan")
async def research_evaluation_plan_api(research: AuditResearchAgent = Depends(get_research_agent)):
    return {"success": True, "plan": research.evaluation_plan(), "timestamp": datetime.now().isoformat()}


@app.get("/api/product/overview")
async def product_overview_api():
    rag_stats = rag_pipeline.get_statistics() if rag_pipeline is not None else {"total_documents": 0}
    return {"success": True, "overview": product_insights.overview(rag_stats), "timestamp": datetime.now().isoformat()}


@app.get("/api/product/risk-register")
async def product_risk_register_api():
    return {"success": True, "risks": product_insights.risk_register(), "timestamp": datetime.now().isoformat()}


@app.get("/api/product/evidence-requests")
async def product_evidence_requests_api():
    return {"success": True, "requests": product_insights.evidence_requests(), "timestamp": datetime.now().isoformat()}


@app.get("/api/product/control-health")
async def product_control_health_api():
    return {"success": True, "controls": product_insights.control_health(), "timestamp": datetime.now().isoformat()}


@app.get("/api/skills")
async def skills_api():
    return {"success": True, "skills": skill_registry.list_skills(), "timestamp": datetime.now().isoformat()}


@app.get("/api/mcp/tools")
async def mcp_tools_api():
    return {"success": True, "tools": skill_registry.mcp_tools(), "timestamp": datetime.now().isoformat()}


@app.post("/api/skills/{skill_name}/run")
async def skill_run_api(skill_name: str, request: SkillRunRequest):
    try:
        record = skill_registry.execute(skill_name, request.input)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Skill 不存在") from exc
    return {"success": record["status"] == "success", "run": record, "timestamp": datetime.now().isoformat()}


@app.get("/api/skills/runs")
async def skill_runs_api(limit: int = 20):
    return {"success": True, "runs": skill_registry.recent_runs(limit), "timestamp": datetime.now().isoformat()}


@app.delete("/api/skills/runs/{run_id}")
async def skill_run_delete_api(run_id: str):
    if not skill_registry.delete_run(run_id):
        raise HTTPException(status_code=404, detail="Skill 运行日志不存在")
    return {"success": True, "deleted": run_id, "timestamp": datetime.now().isoformat()}


@app.get("/api/skills/metrics")
async def skill_metrics_api(limit: int = 500):
    return {"success": True, "metrics": skill_registry.metrics(limit), "timestamp": datetime.now().isoformat()}


@app.post("/api/safety/check")
async def safety_check_api(request: SafetyGateRequest):
    return {"success": True, "gate": safety_gate.inspect(request.payload, request.stage), "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/tasks")
async def agent_task_create_api(request: AgentTaskRequest):
    context = dict(request.context or {})
    lessons = experience_curator.relevant(request.objective)
    if lessons:
        context["experience_lessons"] = lessons
    task = agent_runtime.create_task(request.objective, context)
    return {"success": task["status"] != "blocked", "task": task, "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/tasks")
async def agent_tasks_api(limit: int = 30):
    return {"success": True, "tasks": agent_runtime.list_tasks(limit), "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/tasks/{task_id}")
async def agent_task_detail_api(task_id: str):
    task = agent_runtime.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Agent 任务不存在")
    return {"success": True, "task": task, "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/tasks/{task_id}/episode")
async def agent_task_episode_api(task_id: str):
    try:
        package = agent_runtime.episode_package(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent 任务不存在") from exc
    return {"success": True, "episode": package, "timestamp": datetime.now().isoformat()}


@app.delete("/api/agent/tasks/{task_id}")
async def agent_task_delete_api(task_id: str):
    if not agent_runtime.delete_task(task_id):
        raise HTTPException(status_code=404, detail="Agent 任务不存在")
    return {"success": True, "deleted": task_id, "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/tasks/{task_id}/run-next")
async def agent_task_run_next_api(task_id: str):
    try:
        task = agent_runtime.run_next_step(task_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent 任务不存在") from exc
    return {"success": task["status"] != "blocked", "task": task, "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/tasks/{task_id}/run")
async def agent_task_run_api(task_id: str, request: AgentLoopRequest):
    try:
        task = agent_runtime.run_until_pause(task_id, request.max_steps)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent 任务不存在") from exc
    return {
        "success": task["status"] != "blocked",
        "task": task,
        "loop": task.get("loop", {}),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/agent/tasks/{task_id}/evaluate")
async def agent_task_evaluate_api(task_id: str):
    try:
        report = component_evaluator.evaluate_task(task_id, persist=True)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent 任务不存在") from exc
    return {"success": True, "evaluation": report, "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/tasks/{task_id}/evidence-graph")
async def agent_task_evidence_graph_api(task_id: str):
    try:
        report = component_evaluator.evaluate_task(task_id, persist=False)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent 任务不存在") from exc
    return {
        "success": True,
        "graph": report["evidence_graph"],
        "trace_binding": report["trace_binding"],
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/agent/tasks/{task_id}/curate")
async def agent_task_curate_api(task_id: str):
    task = agent_runtime.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Agent 任务不存在")
    evaluation = component_evaluator.evaluate_task(task_id, persist=True)
    experience = experience_curator.propose(task, evaluation)
    return {
        "success": True,
        "experience": experience,
        "evaluation": evaluation,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/agent/experience")
async def agent_experience_api(limit: int = 30, status: Optional[str] = None):
    return {
        "success": True,
        "experiences": experience_curator.list(limit=limit, status=status),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/agent/experience/{experience_id}/review")
async def agent_experience_review_api(experience_id: str, request: ExperienceReviewRequest):
    try:
        experience = experience_curator.review(
            experience_id,
            request.decision,
            request.reviewer,
            request.comment,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="经验候选不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"success": True, "experience": experience, "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/tasks/{task_id}/steps")
async def agent_task_add_step_api(task_id: str, request: AgentTaskStepRequest):
    try:
        task = agent_runtime.add_step(task_id, request.model_dump())
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Agent 任务不存在") from exc
    return {"success": True, "task": task, "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/observability")
async def agent_observability_api():
    return {"success": True, "observability": agent_runtime.observability(), "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/evolution")
async def agent_evolution_api():
    return {"success": True, "evolution": evolution_harness.report(), "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/harness")
async def agent_harness_api():
    return {"success": True, "harness": harness_control.summary(), "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/harness/candidates/{candidate_id}/evaluate")
async def harness_candidate_evaluate_api(candidate_id: str, request: HarnessEvaluationRequest):
    try:
        candidate = harness_control.evaluate_candidate(
            candidate_id,
            request.baseline,
            request.held_in,
            request.held_out,
            request.checks,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Harness 候选不存在") from exc
    return {"success": True, "candidate": candidate, "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/harness/candidates/{candidate_id}/review")
async def harness_candidate_review_api(candidate_id: str, request: HarnessReviewRequest):
    try:
        candidate = harness_control.review_candidate(candidate_id, request.decision, request.reviewer, request.comment)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Harness 候选不存在") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"success": True, "candidate": candidate, "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/harness/candidates/{candidate_id}/evaluate-runs")
async def harness_candidate_evaluate_runs_api(candidate_id: str, request: HarnessRunEvaluationRequest):
    try:
        candidate = harness_control.evaluate_from_runs(
            candidate_id,
            evaluation_repository,
            request.baseline_run_id,
            request.held_in_run_id,
            request.held_out_run_id,
            request.checks,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"Harness 候选或评测运行不存在：{exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"success": True, "candidate": candidate, "timestamp": datetime.now().isoformat()}


@app.delete("/api/agent/harness/candidates/{candidate_id}")
async def harness_candidate_archive_api(candidate_id: str):
    if not harness_control.archive_candidate(candidate_id):
        raise HTTPException(status_code=404, detail="Harness 候选不存在")
    return {"success": True, "archived": candidate_id, "recoverable": True, "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/quality-diagnostics")
async def agent_quality_diagnostics_api():
    rag_stats = rag_pipeline.get_statistics() if rag_pipeline is not None else {"total_documents": 0, "total_chunks": 0}
    return {"success": True, "diagnostics": agent_quality.report(rag_stats), "timestamp": datetime.now().isoformat()}


@app.get("/api/agent/evolution/market")
async def agent_evolution_market_api():
    report = evolution_harness.report()
    return {
        "success": True,
        "market_radar": report.get("market_radar", {}),
        "trajectory_protocol": report.get("trajectory_protocol", {}),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/agent/evolution/proposals/{proposal_id}/task")
async def agent_evolution_proposal_task_api(proposal_id: str):
    try:
        task = evolution_harness.create_task_from_proposal(proposal_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Evolution proposal not found") from exc
    return {"success": True, "task": task, "timestamp": datetime.now().isoformat()}


@app.post("/api/agent/route")
async def agent_route_preview_api(request: RoutePreviewRequest):
    return {"success": True, "routing": intent_router.classify(request.message), "timestamp": datetime.now().isoformat()}


@app.get("/api/memory/sessions")
async def memory_sessions_api(limit: int = 20):
    return {
        "success": True,
        "sessions": conversation_memory.list_sessions(limit),
        "stats": conversation_memory.stats(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/memory/sessions/{session_id}")
async def memory_session_detail_api(session_id: str):
    session = conversation_memory.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="会话记忆不存在")
    return {"success": True, "session": session, "timestamp": datetime.now().isoformat()}


@app.delete("/api/memory/sessions/{session_id}")
async def memory_session_delete_api(session_id: str):
    if not conversation_memory.delete_session(session_id):
        raise HTTPException(status_code=404, detail="会话记忆不存在")
    return {"success": True, "deleted": session_id, "timestamp": datetime.now().isoformat()}


@app.get("/api/search")
async def global_search_api(q: str = "", limit: int = 12):
    return {"success": True, "query": q, "results": collect_search_results(q, limit), "timestamp": datetime.now().isoformat()}


@app.get("/api/audit/runs")
async def audit_runs_api(limit: int = 20):
    return {"success": True, "runs": audit_repository.list_runs(limit=limit), "timestamp": datetime.now().isoformat()}


@app.get("/api/audit/runs/{run_id}")
async def audit_run_detail_api(run_id: str):
    record = audit_repository.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


@app.delete("/api/audit/runs/{run_id}")
async def audit_run_delete_api(run_id: str):
    if not audit_repository.delete_run(run_id):
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    return {"success": True, "deleted": run_id, "timestamp": datetime.now().isoformat()}


@app.post("/api/audit/runs/{run_id}/review")
async def audit_run_review_api(run_id: str, request: ReviewRequest):
    record = audit_repository.add_review(run_id, request.reviewer, request.decision, request.comment)
    if not record:
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


@app.post("/api/audit/runs/{run_id}/tasks/{task_id}")
async def audit_task_update_api(run_id: str, task_id: str, request: TaskUpdateRequest):
    status_map = {"todo": "未开始", "doing": "进行中", "verifying": "待验证", "done": "已完成", "closed": "已关闭"}
    record = audit_repository.update_task(run_id, task_id, status_map.get(request.status, request.status), request.owner, request.note)
    if not record:
        raise HTTPException(status_code=404, detail="审计运行记录或任务不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


@app.post("/api/audit/runs/{run_id}/evidence/{request_id}")
async def audit_evidence_update_api(run_id: str, request_id: str, request: EvidenceUpdateRequest):
    status_map = {"todo": "待收集", "received": "已收到", "need_more": "需补充", "verified": "已验证", "na": "不适用"}
    record = audit_repository.update_evidence_request(run_id, request_id, status_map.get(request.status, request.status), request.owner, request.note)
    if not record:
        raise HTTPException(status_code=404, detail="审计运行记录或证据请求不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


@app.post("/api/audit/runs/{run_id}/controls/{control_id}/test")
async def audit_control_test_update_api(run_id: str, control_id: str, request: ControlTestUpdateRequest):
    result_map = {"pending": "待执行", "pass": "通过", "exception": "例外", "na": "不适用", "expand": "需扩大样本"}
    record = audit_repository.update_control_test(run_id, control_id, result_map.get(request.result, request.result), request.tester, request.exception)
    if not record:
        raise HTTPException(status_code=404, detail="审计运行记录或控制测试不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


@app.post("/api/audit/runs/{run_id}/evidence-analyses")
async def audit_attach_evidence_analysis_api(run_id: str, request: EvidenceAttachRequest):
    analysis = evidence_analyzer.get_analysis(request.analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="证据分析记录不存在")
    record = audit_repository.attach_evidence_analysis(run_id, analysis)
    if not record:
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


@app.get("/api/audit/runs/{run_id}/report.md", response_class=PlainTextResponse)
async def audit_run_report_api(run_id: str):
    report = audit_repository.render_markdown_report(run_id)
    if report is None:
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    return PlainTextResponse(
        report,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{run_id}.md"'},
    )


@app.get("/api/audit/runs/{run_id}/delivery")
async def audit_delivery_package_api(run_id: str):
    package = audit_delivery.build_package(run_id)
    if package is None:
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    return {"success": True, "package": package, "timestamp": datetime.now().isoformat()}


@app.get("/api/audit/runs/{run_id}/delivery.md", response_class=PlainTextResponse)
async def audit_delivery_markdown_api(run_id: str):
    package = audit_delivery.build_package(run_id)
    if package is None:
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    lines = [
        f"# 审计交付包 - {run_id}",
        "",
        "## 项目信息",
        "",
        f"- 审计对象：{package['engagement'].get('audit_item')}",
        f"- 审计类型：{package['engagement'].get('audit_type')}",
        f"- 参考标准：{package['engagement'].get('standard')}",
        f"- 当前状态：{package['engagement'].get('status')}",
        f"- 项目阶段：{package['engagement'].get('lifecycle_stage')}",
        f"- 审计期间：{package['engagement'].get('audit_period') or ''}",
        f"- 业务背景：{package['engagement'].get('business_context') or ''}",
        f"- 审计范围：{package['engagement'].get('audit_scope') or ''}",
        f"- 重点问题：{package['engagement'].get('key_questions') or ''}",
        f"- 已有证据：{package['engagement'].get('existing_evidence') or ''}",
        "",
        "## 底稿索引",
        "",
        "| 索引 | 名称 | 来源 | 责任人 |",
        "| --- | --- | --- | --- |",
    ]
    for item in package["workpaper_index"]:
        lines.append(f"| {item['ref']} | {item['name']} | {item['source']} | {item['owner']} |")
    lines.extend(["", "## 证据文件分析", ""])
    if not package.get("evidence_analysis_index"):
        lines.append("暂无已归档的证据文件分析。")
    else:
        lines.extend(["| 分析编号 | 文件 | 风险信号 | 映射控制 | 质量门 |", "| --- | --- | --- | --- | --- |"])
        for item in package.get("evidence_analysis_index", []):
            gate = item.get("quality_gate", {})
            lines.append(
                f"| {item.get('analysis_id')} | {item.get('file_name')} | {item.get('risk_count', 0)} | "
                f"{item.get('control_count', 0)} | {gate.get('status', '')} / {gate.get('confidence', '')} |"
            )
    lines.extend(["", "## 证据请求清单", "", "| ID | 来源 | 摘要 | 用途 | 责任人 | 状态 |", "| --- | --- | --- | --- | --- | --- |"])
    for item in package["evidence_request_list"]:
        lines.append(f"| {item['id']} | {item['source']} | {item['summary']} | {item['usage']} | {item.get('owner', '')} | {item['status']} |")
    lines.extend(["", "## 控制测试计划", "", "| 控制 | 领域 | 认定 | 底稿 | 测试程序 | 结果 |", "| --- | --- | --- | --- | --- | --- |"])
    for item in package["control_test_plan"]:
        procedure = str(item.get("test_procedure") or item.get("procedure") or "").replace("|", "/")
        lines.append(f"| {item['control_id']} | {item['domain']} | {item.get('assertion', '')} | {item.get('workpaper_ref', '')} | {procedure} | {item.get('result', '')} |")
    lines.extend(["", "## 访谈计划", "", "| 主题 | 访谈对象 | 关键问题 |", "| --- | --- | --- |"])
    for item in package.get("interview_plan", []):
        lines.append(f"| {item['topic']} | {item['interviewee']} | {'；'.join(item.get('questions', []))} |")
    lines.extend(["", "## 现场工作日程", "", "| 日期 | 活动 | 负责人 | 产出 |", "| --- | --- | --- | --- |"])
    for item in package.get("fieldwork_calendar", []):
        lines.append(f"| {item['day']} | {item['activity']} | {item['owner']} | {item['output']} |")
    lines.extend(["", "## 发现跟踪", ""])
    if not package["finding_tracker"]:
        lines.append("当前未形成重大审计发现。")
    for item in package["finding_tracker"]:
        lines.extend([f"### {item['finding_id']} {item['title']}", "", f"- 严重程度：{item['severity']}", f"- 现状：{item['condition']}", f"- 建议：{item['recommendation']}", ""])
    lines.extend(["", "## 事件轨迹", ""])
    for event in package.get("event_log", []):
        lines.append(f"- {event.get('at')} / {event.get('type')} / {event.get('message')}")
    return PlainTextResponse("\n".join(lines), media_type="text/markdown; charset=utf-8")


@app.post("/api/knowledge/add")
async def add_knowledge_api(request: KnowledgeRequest, rag=Depends(get_rag_pipeline)):
    result = rag.add_knowledge(request.text, request.metadata)
    return {"success": True, "result": result, "timestamp": datetime.now().isoformat()}


@app.post("/api/knowledge/upload")
async def upload_knowledge_file(file: UploadFile = File(...), rag=Depends(get_rag_pipeline)):
    try:
        upload = await read_validated_upload(
            file,
            allowed_extensions={".txt", ".md", ".csv", ".json", ".log"},
            max_bytes=int(UPLOAD_CONFIG["knowledge_max_bytes"]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if upload.security["prompt_injection_detected"] and UPLOAD_CONFIG["reject_prompt_injection"]:
        raise HTTPException(status_code=422, detail="文档包含 Prompt Injection 风险指令，已拒绝进入知识库")
    if upload.security["credential_like_content_detected"]:
        raise HTTPException(status_code=422, detail="文档疑似包含凭据，请脱敏后重新上传")
    tenant_dir = PATHS["uploads"] / current_tenant_id()
    tenant_dir.mkdir(parents=True, exist_ok=True)
    target = tenant_dir / f"{upload.sha256[:20]}{upload.suffix}"
    target.write_bytes(upload.content)
    result = rag.add_file(
        str(target),
        {
            "original_file_name": upload.original_name,
            "sha256": upload.sha256,
            "size_bytes": upload.size_bytes,
            "upload_security": upload.security,
        },
    )
    return {
        "success": True,
        "file": upload.original_name,
        "sha256": upload.sha256,
        "size_bytes": upload.size_bytes,
        "security": upload.security,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/evidence/analyze")
async def analyze_evidence_api(
    file: UploadFile = File(...),
    audit_item: str = Form(""),
    audit_type: str = Form(""),
    standard_type: str = Form(""),
):
    try:
        upload = await read_validated_upload(
            file,
            allowed_extensions={".txt", ".md", ".csv", ".tsv", ".json", ".log"},
            max_bytes=int(UPLOAD_CONFIG["evidence_max_bytes"]),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if upload.security["prompt_injection_detected"] and UPLOAD_CONFIG["reject_prompt_injection"]:
        raise HTTPException(status_code=422, detail="证据文档包含 Prompt Injection 风险指令，已隔离")
    if upload.security["credential_like_content_detected"]:
        raise HTTPException(status_code=422, detail="证据文档疑似包含有效凭据，请先完成脱敏")
    result = evidence_analyzer.analyze_file(
        upload.original_name,
        upload.content,
        {
            "audit_item": audit_item,
            "audit_type": audit_type,
            "standard_type": standard_type,
            "sha256": upload.sha256,
            "size_bytes": upload.size_bytes,
            "upload_security": upload.security,
        },
    )
    return {"success": True, "analysis": result, "timestamp": datetime.now().isoformat()}


@app.get("/api/evidence/analyses")
async def evidence_analyses_api(limit: int = 20):
    return {"success": True, "analyses": evidence_analyzer.list_analyses(limit), "timestamp": datetime.now().isoformat()}


@app.get("/api/evidence/analyses/{analysis_id}")
async def evidence_analysis_detail_api(analysis_id: str):
    record = evidence_analyzer.get_analysis(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="证据分析记录不存在")
    return {"success": True, "analysis": record, "timestamp": datetime.now().isoformat()}


@app.delete("/api/evidence/analyses/{analysis_id}")
async def evidence_analysis_delete_api(analysis_id: str):
    if not evidence_analyzer.delete_analysis(analysis_id):
        raise HTTPException(status_code=404, detail="证据分析记录不存在")
    return {"success": True, "deleted": analysis_id, "timestamp": datetime.now().isoformat()}


@app.get("/api/knowledge/query")
async def query_knowledge_api(question: str, context: Optional[str] = None, rag=Depends(get_rag_pipeline)):
    context_dict = None
    if context:
        try:
            context_dict = json.loads(context)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="context 必须是 JSON 字符串") from exc
    result = rag.query(question, context_dict)
    return {"success": True, "question": question, "result": result, "timestamp": datetime.now().isoformat()}


@app.post("/api/knowledge/build")
async def build_knowledge_graph_api(text: str = Form(...), language: str = Form("auto"), builder: KnowledgeGraphBuilder = Depends(get_kg_builder)):
    result = builder.build_from_text(text, language)
    return {"success": True, "result": result, "timestamp": datetime.now().isoformat()}


@app.get("/api/knowledge/stats")
async def knowledge_stats_api(rag=Depends(get_rag_pipeline)):
    return {"success": True, "stats": rag.get_statistics(), "timestamp": datetime.now().isoformat()}


@app.post("/api/training/evaluate")
async def evaluate_model_api(request: EvaluationRequest, benchmark=Depends(get_evaluator)):
    test_cases = request.test_cases or benchmark.create_test_cases()
    key = cache_key("agent_eval", {"model_path": request.model_path, "test_cases": test_cases})
    if key in evaluation_cache:
        results = clone_payload(evaluation_cache[key])
        results["cache_hit"] = True
    else:
        results = benchmark.evaluate_agent(test_cases)
        evaluation_cache[key] = clone_payload(results)
        results["cache_hit"] = False
    run = evaluation_repository.create_run("agent", request.model_dump(), results)
    return {"success": True, "run": run, "results": results, "timestamp": datetime.now().isoformat()}


@app.get("/api/evaluation/components")
async def evaluation_components_api():
    return {
        "success": True,
        "catalog": component_evaluator.catalog(),
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/evaluation/runtime")
async def evaluate_runtime_components_api(limit: int = 12):
    report = component_evaluator.evaluate_runtime(limit=limit, persist=True)
    return {
        "success": True,
        "evaluation": report,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/evaluation/rag")
async def evaluate_rag_api(request: RAGEvaluationRequest, rag=Depends(get_rag_pipeline)):
    key = cache_key("rag_eval", {"cases": request.cases})
    if key in evaluation_cache:
        results = clone_payload(evaluation_cache[key])
        results["cache_hit"] = True
    else:
        results = RAGEvaluator(rag).evaluate(request.cases)
        evaluation_cache[key] = clone_payload(results)
        results["cache_hit"] = False
    run = evaluation_repository.create_run("rag", request.model_dump(), results)
    return {"success": True, "run": run, "results": results, "timestamp": datetime.now().isoformat()}


@app.get("/api/evaluation/runs")
async def evaluation_runs_api(limit: int = 20):
    return {"success": True, "runs": evaluation_repository.list_runs(limit), "timestamp": datetime.now().isoformat()}


@app.get("/api/evaluation/runs/{run_id}")
async def evaluation_run_detail_api(run_id: str):
    record = evaluation_repository.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


@app.delete("/api/evaluation/runs/{run_id}")
async def evaluation_run_delete_api(run_id: str):
    if not evaluation_repository.delete_run(run_id):
        raise HTTPException(status_code=404, detail="评测记录不存在")
    return {"success": True, "deleted": run_id, "timestamp": datetime.now().isoformat()}


@app.get("/api/session/history/{session_id}")
async def get_session_history(session_id: str, agent: AuditAgent = Depends(get_audit_agent)):
    persistent = conversation_memory.get_session(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "history": agent.get_session_history(session_id),
        "persistent_memory": persistent,
        "timestamp": datetime.now().isoformat(),
    }


def _readiness_payload() -> tuple[Dict[str, Any], bool]:
    services = {
        "llm": bool(LLM_CONFIG.get("enabled")),
        "mysql": False,
        "neo4j": False,
        "rag": rag_pipeline is not None,
        "rag_documents": 0,
        "agent_runtime": True,
        "evolution_harness": True,
        "skills": len(skill_registry.skills),
        "intent_router": True,
        "memory": conversation_memory.stats(),
        "security_mode": SECURITY_CONFIG["mode"],
        "tenant_isolation": bool(SECURITY_CONFIG["tenant_isolation"]),
        "audit_event_chain": audit_event_store.verify(),
        "transactional_storage": {
            "audit_runs": audit_repository.store.health(),
            "evaluation_runs": evaluation_repository.store.health(),
        },
    }
    if audit_agent is not None:
        services.update(audit_agent.get_service_status())
    if rag_pipeline is not None:
        services["rag_documents"] = rag_pipeline.get_statistics().get("total_documents", 0)
    blockers = []
    if SECURITY_CONFIG["mode"] == "enforced" and not SECURITY_CONFIG.get("api_tokens"):
        blockers.append("SECURITY_MODE=enforced 但未配置 API token")
    if not services["audit_event_chain"].get("valid"):
        blockers.append("审计事件哈希链校验失败")
    for name, health in services["transactional_storage"].items():
        if not health.get("ready"):
            blockers.append(f"事务存储不可用：{name}")
    for name in ("data", "uploads", "evaluation_runs", "agent_runtime"):
        path = PATHS[name]
        if not path.exists() or not path.is_dir():
            blockers.append(f"运行目录不可用：{name}")
    return (
        {
            "status": "ready" if not blockers else "not_ready",
            "timestamp": datetime.now().isoformat(),
            "version": "4.1.0",
            "services": services,
            "blockers": blockers,
        },
        not blockers,
    )


@app.get("/api/health/live")
async def health_live_api():
    return {"status": "alive", "timestamp": datetime.now().isoformat(), "version": "4.1.0"}


@app.get("/api/health/ready")
async def health_ready_api():
    payload, ready = _readiness_payload()
    return JSONResponse(content=payload, status_code=200 if ready else 503)


@app.get("/api/health")
async def health_check():
    payload, ready = _readiness_payload()
    return JSONResponse(content=payload, status_code=200 if ready else 503)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_CONFIG["host"], port=WEB_CONFIG["port"])

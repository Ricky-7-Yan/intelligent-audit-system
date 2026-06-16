"""FastAPI web application for the Intelligent Audit System."""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from agents.audit_agent import AuditAgent, CONTROL_LIBRARY
from config import LLM_CONFIG, PATHS, WEB_CONFIG
from knowledge_graph.builder import KnowledgeGraphBuilder
from services.audit_delivery import AuditDeliveryService
from services.audit_repository import AuditRunRepository
from services.audit_templates import list_audit_templates
from services.evaluation_repository import EvaluationRunRepository
from services.evidence_analyzer import EvidenceAnalyzer
from services.product_insights import ProductInsights
from services.rag_evaluator import RAGEvaluator
from services.research_agent import AuditResearchAgent
from services.skill_registry import SkillRegistry


logger = logging.getLogger(__name__)

audit_agent: Optional[AuditAgent] = None
rag_pipeline = None
kg_builder: Optional[KnowledgeGraphBuilder] = None
evaluator = None
audit_repository = AuditRunRepository()
skill_registry = SkillRegistry()
product_insights = ProductInsights(audit_repository, skill_registry)
audit_delivery = AuditDeliveryService(audit_repository)
evaluation_repository = EvaluationRunRepository()
evidence_analyzer = EvidenceAnalyzer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
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
    version="2.9.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=WEB_CONFIG["cors_origins"] or ["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(PATHS["static"])), name="static")
templates = Jinja2Templates(directory=str(PATHS["templates"]))


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


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
        audit_agent = AuditAgent(rag_pipeline=init_rag_lazy())
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
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    return templates.TemplateResponse("audit.html", {"request": request})


@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(request: Request):
    return templates.TemplateResponse("knowledge.html", {"request": request})


@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    return templates.TemplateResponse("training.html", {"request": request})


@app.get("/skills", response_class=HTMLResponse)
async def skills_page(request: Request):
    return templates.TemplateResponse("skills.html", {"request": request})


@app.post("/api/chat")
async def chat_api(request: ChatRequest, agent: AuditAgent = Depends(get_audit_agent)):
    session_id = request.session_id or str(uuid.uuid4())
    result = agent.process_audit_query(request.message, session_id=session_id)
    return {"success": True, "session_id": session_id, "timestamp": datetime.now().isoformat(), **result}


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
            "agent_architecture": ["任务规划", "Skill 注册与执行", "MCP 风格工具描述", "工具调用", "Agentic RAG", "质量门", "人工复核闭环"],
            "rag": ["混合检索", "查询扩展", "来源引用", "降级检索", "RAG 评测"],
            "engineering": ["FastAPI", "持久化审计档案", "报告导出", "健康检查", "Docker 部署"],
            "audit_business": ["审计程序", "抽样计划", "证据请求中心", "控制测试工作台", "审计发现草稿", "整改任务跟踪"],
        },
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/research/answer")
async def research_answer_api(request: ResearchRequest, research: AuditResearchAgent = Depends(get_research_agent)):
    result = research.answer(request.question, request.context)
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


@app.get("/api/audit/runs")
async def audit_runs_api(limit: int = 20):
    return {"success": True, "runs": audit_repository.list_runs(limit=limit), "timestamp": datetime.now().isoformat()}


@app.get("/api/audit/runs/{run_id}")
async def audit_run_detail_api(run_id: str):
    record = audit_repository.get_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="审计运行记录不存在")
    return {"success": True, "run": record, "timestamp": datetime.now().isoformat()}


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
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".csv", ".json", ".log"}:
        raise HTTPException(status_code=400, detail="当前上传接口支持 .txt、.md、.csv、.json、.log 文本文件")
    target = PATHS["uploads"] / f"{uuid.uuid4().hex}{suffix}"
    content = await file.read()
    target.write_bytes(content)
    result = rag.add_file(str(target))
    return {"success": True, "file": file.filename, "result": result, "timestamp": datetime.now().isoformat()}


@app.post("/api/evidence/analyze")
async def analyze_evidence_api(
    file: UploadFile = File(...),
    audit_item: str = Form(""),
    audit_type: str = Form(""),
    standard_type: str = Form(""),
):
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".txt", ".md", ".csv", ".tsv", ".json", ".log"}:
        raise HTTPException(status_code=400, detail="当前证据分析支持 .txt、.md、.csv、.tsv、.json、.log 文件")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="证据文件超过 5MB，请先拆分或抽样上传")
    result = evidence_analyzer.analyze_file(
        file.filename or "evidence",
        content,
        {"audit_item": audit_item, "audit_type": audit_type, "standard_type": standard_type},
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
    results = benchmark.evaluate_agent(test_cases)
    run = evaluation_repository.create_run("agent", request.model_dump(), results)
    return {"success": True, "run": run, "results": results, "timestamp": datetime.now().isoformat()}


@app.post("/api/evaluation/rag")
async def evaluate_rag_api(request: RAGEvaluationRequest, rag=Depends(get_rag_pipeline)):
    results = RAGEvaluator(rag).evaluate(request.cases)
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


@app.get("/api/session/history/{session_id}")
async def get_session_history(session_id: str, agent: AuditAgent = Depends(get_audit_agent)):
    return {"success": True, "session_id": session_id, "history": agent.get_session_history(session_id), "timestamp": datetime.now().isoformat()}


@app.get("/api/health")
async def health_check():
    services = {"llm": bool(LLM_CONFIG.get("enabled")), "mysql": False, "neo4j": False, "rag": rag_pipeline is not None, "rag_documents": 0}
    if audit_agent is not None:
        services.update(audit_agent.get_service_status())
    if rag_pipeline is not None:
        services["rag_documents"] = rag_pipeline.get_statistics().get("total_documents", 0)
    return JSONResponse(content={"status": "healthy", "timestamp": datetime.now().isoformat(), "version": "2.9.0", "services": services})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_CONFIG["host"], port=WEB_CONFIG["port"])

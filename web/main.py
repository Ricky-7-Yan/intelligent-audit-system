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
from config import PATHS, WEB_CONFIG
from knowledge_graph.builder import KnowledgeGraphBuilder
from services.audit_repository import AuditRunRepository


logger = logging.getLogger(__name__)

audit_agent: Optional[AuditAgent] = None
rag_pipeline = None
kg_builder: Optional[KnowledgeGraphBuilder] = None
evaluator = None
audit_repository = AuditRunRepository()


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
    title="智能审计 Agent 平台",
    description="面向审计场景的 Agentic RAG、风险评估和合规分析系统",
    version="2.2.0",
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


class KnowledgeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=200000)
    metadata: Optional[Dict[str, Any]] = None


class EvaluationRequest(BaseModel):
    model_path: str = "current-agent"
    test_cases: Optional[List[Dict[str, Any]]] = None


class ReviewRequest(BaseModel):
    reviewer: str = "复核人"
    decision: str = Field(..., pattern="^(approve|reject|need_evidence)$")
    comment: str = Field("", max_length=4000)


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


@app.post("/api/chat")
async def chat_api(request: ChatRequest, agent: AuditAgent = Depends(get_audit_agent)):
    session_id = request.session_id or str(uuid.uuid4())
    result = agent.process_audit_query(request.message, session_id=session_id)
    return {"success": True, "session_id": session_id, "timestamp": datetime.now().isoformat(), **result}


@app.post("/api/audit")
async def audit_api(request: AuditRequest, agent: AuditAgent = Depends(get_audit_agent)):
    audit_query = f"请对{request.audit_item}进行{request.audit_type}"
    if request.standard_type:
        audit_query += f"，参考{request.standard_type}标准"
    if request.risk_level:
        audit_query += f"，关注{request.risk_level}风险"
    result = agent.process_audit_query(audit_query)
    run = audit_repository.create_run(request.model_dump(), result)
    return {
        "success": True,
        "run_id": run["run_id"],
        "audit_item": request.audit_item,
        "audit_type": request.audit_type,
        "result": result,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/audit/controls")
async def audit_controls_api():
    return {"success": True, "controls": CONTROL_LIBRARY, "timestamp": datetime.now().isoformat()}


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
async def build_knowledge_graph_api(
    text: str = Form(...),
    language: str = Form("auto"),
    builder: KnowledgeGraphBuilder = Depends(get_kg_builder),
):
    result = builder.build_from_text(text, language)
    return {"success": True, "result": result, "timestamp": datetime.now().isoformat()}


@app.get("/api/knowledge/stats")
async def knowledge_stats_api(rag=Depends(get_rag_pipeline)):
    return {"success": True, "stats": rag.get_statistics(), "timestamp": datetime.now().isoformat()}


@app.post("/api/training/evaluate")
async def evaluate_model_api(request: EvaluationRequest, benchmark=Depends(get_evaluator)):
    test_cases = request.test_cases or benchmark.create_test_cases()
    results = benchmark.evaluate_agent(test_cases)
    return {"success": True, "results": results, "timestamp": datetime.now().isoformat()}


@app.get("/api/session/history/{session_id}")
async def get_session_history(session_id: str, agent: AuditAgent = Depends(get_audit_agent)):
    return {
        "success": True,
        "session_id": session_id,
        "history": agent.get_session_history(session_id),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/health")
async def health_check(agent: AuditAgent = Depends(get_audit_agent), rag=Depends(get_rag_pipeline)):
    return JSONResponse(
        content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "2.2.0",
            "services": {**agent.get_service_status(), "rag_documents": rag.get_statistics().get("total_documents", 0)},
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=WEB_CONFIG["host"], port=WEB_CONFIG["port"])

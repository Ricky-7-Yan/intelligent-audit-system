"""
智能审计决策系统Web界面
Intelligent Audit Decision System Web Interface
"""

from fastapi import FastAPI, Request, Form, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
from datetime import datetime
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

# 导入自定义模块
from agents.audit_agent import AuditAgent
from knowledge_graph.builder import KnowledgeGraphBuilder
from config import WEB_CONFIG, PATHS

# 延迟导入RAG和训练模块，避免启动时下载模型
RAGPipeline = None
BenchmarkEvaluator = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 生命周期事件
@asynccontextmanager
async def lifespan(app: FastAPI):
    """生命周期管理"""
    # 启动事件
    logger.info("智能审计决策系统启动中...")

    # 确保目录存在
    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)

    logger.info("系统启动完成")

    yield

    # 关闭事件
    logger.info("智能审计决策系统关闭中...")

    # 关闭全局资源
    global audit_agent, rag_pipeline, kg_builder
    if audit_agent:
        audit_agent.close()
    if kg_builder:
        kg_builder.close()

    logger.info("系统关闭完成")

# 创建FastAPI应用
app = FastAPI(
    title="智能审计决策系统",
    description="基于LLM的智能审计决策系统",
    version="1.0.0",
    lifespan=lifespan
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件
app.mount("/static", StaticFiles(directory=str(PATHS['static'])), name="static")

# 设置模板
templates = Jinja2Templates(directory=str(PATHS['templates']))

# 全局变量
audit_agent = None
rag_pipeline = None
kg_builder = None
evaluator = None

# 延迟初始化RAG，避免启动时下载模型
def init_rag_lazy():
    """延迟初始化RAG"""
    global rag_pipeline, RAGPipeline
    if rag_pipeline is None:
        try:
            # 动态导入
            from rag.agentic_rag import RAGPipeline as _RAGPipeline
            RAGPipeline = _RAGPipeline
            rag_pipeline = RAGPipeline()
        except Exception as e:
            logger.error(f"RAG初始化失败: {e}")
            rag_pipeline = None
    return rag_pipeline

# 请求模型
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class AuditRequest(BaseModel):
    audit_item: str
    audit_type: str
    standard_type: Optional[str] = None
    risk_level: Optional[str] = None

class KnowledgeRequest(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class EvaluationRequest(BaseModel):
    model_path: str
    test_cases: Optional[List[Dict[str, Any]]] = None

# 依赖注入
def get_audit_agent():
    global audit_agent
    if audit_agent is None:
        audit_agent = AuditAgent()
    return audit_agent

def get_rag_pipeline():
    """获取RAG管道（延迟初始化）"""
    return init_rag_lazy()

def get_kg_builder():
    global kg_builder
    if kg_builder is None:
        kg_builder = KnowledgeGraphBuilder()
    return kg_builder

def get_evaluator():
    """获取评估器（延迟初始化）"""
    global evaluator, BenchmarkEvaluator
    if evaluator is None:
        try:
            from training.training_pipeline import BenchmarkEvaluator as _BenchmarkEvaluator
            BenchmarkEvaluator = _BenchmarkEvaluator
            evaluator = BenchmarkEvaluator()
        except Exception as e:
            logger.error(f"评估器初始化失败: {e}")
            evaluator = None
    return evaluator

# 路由
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """主页"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def chat_page(request: Request):
    """聊天页面"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/audit", response_class=HTMLResponse)
async def audit_page(request: Request):
    """审计页面"""
    return templates.TemplateResponse("audit.html", {"request": request})

@app.get("/knowledge", response_class=HTMLResponse)
async def knowledge_page(request: Request):
    """知识管理页面"""
    return templates.TemplateResponse("knowledge.html", {"request": request})

@app.get("/training", response_class=HTMLResponse)
async def training_page(request: Request):
    """训练页面"""
    return templates.TemplateResponse("training.html", {"request": request})

@app.post("/api/chat")
async def chat_api(request: ChatRequest, agent: AuditAgent = Depends(get_audit_agent)):
    """聊天API"""
    try:
        session_id = request.session_id or str(uuid.uuid4())

        result = agent.process_audit_query(
            request.message,
            session_id=session_id
        )

        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "response": result["response"],
            "audit_context": result.get("audit_context", {}),
            "risk_assessment": result.get("risk_assessment"),
            "compliance_check": result.get("compliance_check"),
            "recommendations": result.get("recommendations", []),
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"聊天API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/audit", response_class=JSONResponse)
async def audit_api(request: AuditRequest, agent: AuditAgent = Depends(get_audit_agent)):
    """审计API"""
    try:
        # 构建审计查询
        audit_query = f"请对{request.audit_item}进行{request.audit_type}审计"

        if request.standard_type:
            audit_query += f"，参考{request.standard_type}标准"

        if request.risk_level:
            audit_query += f"，风险等级为{request.risk_level}"

        result = agent.process_audit_query(audit_query)

        return JSONResponse(content={
            "success": True,
            "audit_item": request.audit_item,
            "audit_type": request.audit_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"审计API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/knowledge/add", response_class=JSONResponse)
async def add_knowledge_api(request: KnowledgeRequest, rag: RAGPipeline = Depends(get_rag_pipeline)):
    """添加知识API"""
    try:
        rag.add_knowledge(request.text, request.metadata)

        return JSONResponse(content={
            "success": True,
            "message": "知识添加成功",
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"添加知识API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/knowledge/query", response_class=JSONResponse)
async def query_knowledge_api(
    question: str,
    context: Optional[str] = None,
    rag: RAGPipeline = Depends(get_rag_pipeline)
):
    """查询知识API"""
    try:
        context_dict = json.loads(context) if context else None

        result = rag.query(question, context_dict)

        return JSONResponse(content={
            "success": True,
            "question": question,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"查询知识API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/knowledge/build", response_class=JSONResponse)
async def build_knowledge_graph_api(
    text: str = Form(...),
    language: str = Form("auto"),
    kg_builder: KnowledgeGraphBuilder = Depends(get_kg_builder)
):
    """构建知识图谱API"""
    try:
        result = kg_builder.build_from_text(text, language)

        return JSONResponse(content={
            "success": True,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"构建知识图谱API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/knowledge/stats", response_class=JSONResponse)
async def knowledge_stats_api(rag: RAGPipeline = Depends(get_rag_pipeline)):
    """知识统计API"""
    try:
        stats = rag.get_statistics()

        return JSONResponse(content={
            "success": True,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"知识统计API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/training/evaluate", response_class=JSONResponse)
async def evaluate_model_api(request: EvaluationRequest, evaluator: BenchmarkEvaluator = Depends(get_evaluator)):
    """模型评估API"""
    try:
        if not request.test_cases:
            test_cases = evaluator.create_test_cases()
        else:
            test_cases = request.test_cases

        # 这里需要加载模型进行评估
        # 简化处理，返回模拟结果
        evaluation_results = {
            "overall_score": 0.85,
            "total_tests": len(test_cases),
            "category_scores": {
                "governance": 0.9,
                "risk_assessment": 0.8,
                "compliance": 0.85,
                "security": 0.9,
                "data_protection": 0.8
            },
            "evaluation_date": datetime.now().isoformat()
        }

        return JSONResponse(content={
            "success": True,
            "results": evaluation_results,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"模型评估API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/session/history/{session_id}", response_class=JSONResponse)
async def get_session_history(session_id: str, agent: AuditAgent = Depends(get_audit_agent)):
    """获取会话历史API"""
    try:
        history = agent.get_session_history(session_id)

        return JSONResponse(content={
            "success": True,
            "session_id": session_id,
            "history": history,
            "timestamp": datetime.now().isoformat()
        })

    except Exception as e:
        logger.error(f"获取会话历史API错误: {e}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/api/health", response_class=JSONResponse)
async def health_check():
    """健康检查API"""
    return JSONResponse(content={
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })

@app.get("/test")
async def test_page():
    """测试页面"""
    return {"message": "系统运行正常！访问 http://localhost:8000 查看主页"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=WEB_CONFIG['host'],
        port=WEB_CONFIG['port']
    )


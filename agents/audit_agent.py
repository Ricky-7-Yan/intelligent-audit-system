"""Enterprise audit Agent orchestration.

The agent follows a practical audit workflow:
scope planning -> RAG evidence retrieval -> control mapping -> risk scoring ->
audit program generation -> quality gate -> findings and remediation planning.
It keeps deterministic fallbacks so the product remains usable without external
LLM, database, graph, or embedding services.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from config import AUDIT_CONFIG, LLM_CONFIG, MYSQL_CONFIG, NEO4J_CONFIG

try:
    import pymysql
except Exception:  # pragma: no cover
    pymysql = None

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None


logger = logging.getLogger(__name__)


AUDIT_STANDARDS: Dict[str, Dict[str, Any]] = {
    "COBIT": {
        "name": "COBIT 2019",
        "focus": "企业 IT 治理、价值交付、风险优化、资源优化和绩效度量",
        "controls": ["治理目标映射", "流程责任矩阵", "绩效指标", "风险场景管理"],
    },
    "ISO27001": {
        "name": "ISO/IEC 27001",
        "focus": "信息安全管理体系、风险评估、控制措施选择和持续改进",
        "controls": ["访问控制", "资产管理", "事件响应", "供应商安全", "备份与恢复"],
    },
    "SOX": {
        "name": "Sarbanes-Oxley Act",
        "focus": "财务报告相关内部控制、ITGC、变更审批、职责分离和审计证据",
        "controls": ["职责分离", "变更管理", "日志留存", "财务数据完整性", "管理层复核"],
    },
    "数据安全法": {
        "name": "数据安全法 / 个人信息保护相关要求",
        "focus": "数据分类分级、重要数据保护、个人信息处理、风险监测和应急处置",
        "controls": ["分类分级", "最小权限", "数据加密", "共享审批", "应急预案"],
    },
}


RISK_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "权限": {"score": 0.84, "risk": "权限滥用、越权访问或职责分离不足", "domain": "访问控制"},
    "账号": {"score": 0.78, "risk": "账号生命周期和特权账号管理不足", "domain": "身份治理"},
    "财务": {"score": 0.86, "risk": "财务数据完整性、审批链路和报表可靠性风险", "domain": "财务内控"},
    "变更": {"score": 0.75, "risk": "系统变更未经过充分审批、测试或上线后复核", "domain": "变更管理"},
    "备份": {"score": 0.69, "risk": "备份不可恢复或恢复目标不清晰", "domain": "业务连续性"},
    "日志": {"score": 0.66, "risk": "审计日志不完整、不可追溯或缺少告警", "domain": "监控审计"},
    "数据": {"score": 0.77, "risk": "敏感数据泄露、过度使用或共享不合规", "domain": "数据安全"},
    "接口": {"score": 0.71, "risk": "接口鉴权、限流、对账和异常处置不足", "domain": "接口安全"},
}


CONTROL_LIBRARY: List[Dict[str, Any]] = [
    {
        "id": "AC-01",
        "domain": "访问控制",
        "keywords": ["权限", "账号", "访问", "ERP", "身份"],
        "objective": "确保用户仅拥有完成岗位职责所需的最小权限。",
        "test_procedure": "抽样检查用户权限、角色授权、审批记录和最近一次权限复核结果。",
        "evidence_required": ["权限清单", "授权审批单", "角色矩阵", "定期复核记录"],
        "standards": ["ISO27001", "SOX", "COBIT"],
    },
    {
        "id": "AC-02",
        "domain": "职责分离",
        "keywords": ["权限", "财务", "职责", "审批", "制单"],
        "objective": "防止同一人员同时拥有发起、审批和复核关键交易的权限。",
        "test_procedure": "识别互斥权限组合，核查例外审批、补偿性控制和整改闭环。",
        "evidence_required": ["职责分离规则", "冲突权限报表", "例外审批", "补偿性控制记录"],
        "standards": ["SOX", "COBIT"],
    },
    {
        "id": "CM-01",
        "domain": "变更管理",
        "keywords": ["变更", "上线", "发布", "系统"],
        "objective": "确保生产变更经过授权、测试、回退设计和上线后复核。",
        "test_procedure": "抽样检查变更单、测试证据、审批链、上线记录和回退方案。",
        "evidence_required": ["变更单", "测试报告", "上线审批", "回退方案", "上线后复核"],
        "standards": ["ISO27001", "SOX", "COBIT"],
    },
    {
        "id": "BC-01",
        "domain": "业务连续性",
        "keywords": ["备份", "恢复", "灾备", "RPO", "RTO"],
        "objective": "确保关键系统和数据可在既定恢复目标内恢复。",
        "test_procedure": "检查备份策略、备份成功率、恢复演练记录和问题整改闭环。",
        "evidence_required": ["备份策略", "备份日志", "恢复演练报告", "RPO/RTO 定义"],
        "standards": ["ISO27001", "COBIT"],
    },
    {
        "id": "LOG-01",
        "domain": "监控审计",
        "keywords": ["日志", "监控", "告警", "审计轨迹"],
        "objective": "确保关键操作可记录、可追溯、可告警。",
        "test_procedure": "检查日志范围、留存周期、防篡改机制、告警规则和处置记录。",
        "evidence_required": ["日志策略", "日志样本", "告警规则", "事件处置单"],
        "standards": ["ISO27001", "数据安全法"],
    },
    {
        "id": "DATA-01",
        "domain": "数据安全",
        "keywords": ["数据", "敏感", "个人信息", "加密", "脱敏"],
        "objective": "确保敏感数据分类分级、授权使用、加密脱敏和共享审批。",
        "test_procedure": "检查数据目录、分类分级、访问授权、加密脱敏和共享审批记录。",
        "evidence_required": ["数据目录", "分类分级规则", "访问授权", "加密配置", "共享审批"],
        "standards": ["数据安全法", "ISO27001"],
    },
    {
        "id": "API-01",
        "domain": "接口安全",
        "keywords": ["接口", "API", "对账", "鉴权", "限流"],
        "objective": "确保接口调用有鉴权、限流、监控、对账和异常处置。",
        "test_procedure": "检查接口台账、密钥管理、调用日志、限流策略和对账记录。",
        "evidence_required": ["接口台账", "鉴权配置", "调用日志", "限流规则", "对账记录"],
        "standards": ["ISO27001", "COBIT"],
    },
]


@dataclass
class ServiceStatus:
    llm: bool = False
    mysql: bool = False
    neo4j: bool = False
    rag: bool = False


class OptionalAuditTools:
    def __init__(self) -> None:
        self.mysql_connection = None
        self.neo4j_driver = None
        self.status = ServiceStatus()
        self._connect_mysql()
        self._connect_neo4j()

    def _connect_mysql(self) -> None:
        if not pymysql or not MYSQL_CONFIG.get("password"):
            return
        try:
            self.mysql_connection = pymysql.connect(**MYSQL_CONFIG)
            self.status.mysql = True
        except Exception as exc:
            logger.info("MySQL unavailable, using built-in standards: %s", exc)

    def _connect_neo4j(self) -> None:
        if not GraphDatabase or not NEO4J_CONFIG.get("password"):
            return
        try:
            self.neo4j_driver = GraphDatabase.driver(
                NEO4J_CONFIG["uri"],
                auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"]),
                connection_timeout=NEO4J_CONFIG.get("timeout", 3),
            )
            self.neo4j_driver.verify_connectivity()
            self.status.neo4j = True
        except Exception as exc:
            logger.info("Neo4j unavailable, using local graph fallback: %s", exc)
            self.neo4j_driver = None

    def close(self) -> None:
        if self.mysql_connection:
            self.mysql_connection.close()
        if self.neo4j_driver:
            self.neo4j_driver.close()

    def query_knowledge_graph(self, query: str) -> List[Dict[str, Any]]:
        if not self.neo4j_driver:
            return []
        cypher = """
        MATCH (n)-[r]->(m)
        WHERE toLower(coalesce(n.text, n.name, '')) CONTAINS toLower($query)
           OR toLower(coalesce(m.text, m.name, '')) CONTAINS toLower($query)
        RETURN coalesce(n.text, n.name) AS source,
               labels(n) AS source_labels,
               type(r) AS relation,
               coalesce(m.text, m.name) AS target,
               labels(m) AS target_labels
        LIMIT 12
        """
        try:
            with self.neo4j_driver.session() as session:
                return [dict(record) for record in session.run(cypher, query=query)]
        except Exception as exc:
            logger.warning("Knowledge graph query failed: %s", exc)
            return []

    def get_standards(self, standard_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if self.mysql_connection:
            try:
                with self.mysql_connection.cursor() as cursor:
                    if standard_type:
                        cursor.execute(
                            """
                            SELECT standard_name, standard_type, version, description, requirements
                            FROM audit_standards
                            WHERE standard_type = %s
                            """,
                            (standard_type,),
                        )
                    else:
                        cursor.execute(
                            """
                            SELECT standard_name, standard_type, version, description, requirements
                            FROM audit_standards
                            """
                        )
                    rows = cursor.fetchall()
                return [
                    {
                        "name": row[0],
                        "type": row[1],
                        "version": row[2],
                        "description": row[3],
                        "requirements": json.loads(row[4]) if row[4] else {},
                    }
                    for row in rows
                ]
            except Exception as exc:
                logger.warning("Audit standards query failed: %s", exc)

        if standard_type:
            standard = AUDIT_STANDARDS.get(standard_type)
            return [{**standard, "type": standard_type}] if standard else []
        return [{**value, "type": key} for key, value in AUDIT_STANDARDS.items()]


class AuditAgent:
    def __init__(self, rag_pipeline: Any = None) -> None:
        self.tools = OptionalAuditTools()
        self.rag_pipeline = rag_pipeline
        self.session_memory: Dict[str, List[BaseMessage]] = {}
        self.llm = self._init_llm()
        logger.info(
            "AuditAgent initialized. LLM=%s MySQL=%s Neo4j=%s",
            bool(self.llm),
            self.tools.status.mysql,
            self.tools.status.neo4j,
        )

    def _init_llm(self) -> Any:
        if os.getenv("AUDIT_DISABLE_LLM", "1").lower() in {"1", "true", "yes"}:
            return None
        if not LLM_CONFIG.get("enabled"):
            return None
        try:
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                api_key=LLM_CONFIG["api_key"],
                base_url=LLM_CONFIG["base_url"],
                model=LLM_CONFIG["model"],
                temperature=LLM_CONFIG["temperature"],
                max_tokens=LLM_CONFIG["max_tokens"],
            )
        except Exception as exc:
            logger.warning("LLM client initialization failed: %s", exc)
            return None

    def process_audit_query(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_memory.setdefault(session_id, [])
        self.session_memory[session_id].append(HumanMessage(content=user_input))
        self._trim_session(session_id)

        trace: List[Dict[str, Any]] = []
        audit_context = self._extract_context(user_input)
        task_plan = self._create_task_plan(audit_context)
        trace.append(self._trace("planner", "generated", f"生成 {len(task_plan)} 个审计任务"))

        retrieved = self._retrieve_context(user_input, audit_context)
        evidence_pack = self._build_evidence_pack(retrieved, audit_context)
        trace.append(self._trace("retriever", "completed", f"形成 {len(evidence_pack)} 条证据线索"))

        control_matrix = self._build_control_matrix(user_input, audit_context, evidence_pack)
        trace.append(self._trace("control_mapper", "completed", f"映射 {len(control_matrix)} 项关键控制"))

        risk_assessment = self._assess_risk(user_input, audit_context, retrieved, control_matrix)
        compliance_check = self._check_compliance(audit_context, risk_assessment, control_matrix)
        audit_program = self._build_audit_program(audit_context, control_matrix, risk_assessment)
        sampling_plan = self._build_sampling_plan(audit_context, control_matrix, risk_assessment)
        findings = self._draft_findings(audit_context, control_matrix, risk_assessment, compliance_check, evidence_pack)
        quality_gate = self._quality_gate(evidence_pack, control_matrix, risk_assessment, compliance_check)
        recommendations = self._generate_recommendations(risk_assessment, compliance_check, control_matrix, quality_gate)
        response = self._compose_response(
            user_input,
            audit_context,
            retrieved,
            risk_assessment,
            compliance_check,
            recommendations,
            quality_gate,
        )
        trace.append(self._trace("quality_gate", quality_gate["status"], f"置信度 {quality_gate['confidence']}"))

        self.session_memory[session_id].append(AIMessage(content=response))
        self._trim_session(session_id)

        return {
            "session_id": session_id,
            "response": response,
            "audit_context": audit_context,
            "task_plan": task_plan,
            "retrieval": retrieved,
            "evidence_pack": evidence_pack,
            "control_matrix": control_matrix,
            "audit_program": audit_program,
            "sampling_plan": sampling_plan,
            "findings": findings,
            "risk_assessment": risk_assessment,
            "compliance_check": compliance_check,
            "quality_gate": quality_gate,
            "recommendations": recommendations,
            "execution_trace": trace,
            "service_status": self.get_service_status(),
        }

    def _trace(self, stage: str, status: str, detail: str) -> Dict[str, Any]:
        return {"stage": stage, "status": status, "detail": detail, "at": datetime.now().isoformat()}

    def _trim_session(self, session_id: str) -> None:
        max_messages = AUDIT_CONFIG["max_session_messages"]
        self.session_memory[session_id] = self.session_memory[session_id][-max_messages:]

    def _extract_context(self, text: str) -> Dict[str, Any]:
        normalized = text.upper()
        standards = [key for key in AUDIT_STANDARDS if key.upper() in normalized]
        if "ISO" in normalized and "ISO27001" not in standards:
            standards.append("ISO27001")

        audit_types = []
        for keyword in ["安全审计", "合规审计", "风险评估", "内部控制审计", "数据审计", "财务审计"]:
            if keyword in text:
                audit_types.append(keyword)

        item = self._guess_audit_item(text)
        topics = [key for key in RISK_KEYWORDS if key in text or key in item]
        return {
            "audit_item": item,
            "audit_types": audit_types or ["综合审计分析"],
            "standards": standards or self._infer_standards(text),
            "key_risk_topics": topics,
            "business_domain": self._business_domain(text, topics),
            "generated_at": datetime.now().isoformat(),
        }

    def _guess_audit_item(self, text: str) -> str:
        patterns = [r"对(.+?)进行", r"评估(.+?)的", r"检查(.+?)是否", r"分析(.+?)的"]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" ，。")
        for keyword in ["ERP系统", "CRM系统", "财务系统", "权限管理", "数据备份", "日志管理", "接口管理"]:
            if keyword in text:
                return keyword
        return "待审计对象"

    def _infer_standards(self, text: str) -> List[str]:
        normalized = text.upper()
        inferred = []
        if any(word in text for word in ["财务", "报表", "凭证"]) or "SOX" in normalized:
            inferred.append("SOX")
        if any(word in text for word in ["安全", "权限", "账号", "日志", "备份", "ACCESS", "SECURITY"]):
            inferred.append("ISO27001")
        if any(word in text for word in ["治理", "系统", "流程"]) or "IT" in normalized:
            inferred.append("COBIT")
        if any(word in text for word in ["数据", "个人信息", "敏感", "DATA"]):
            inferred.append("数据安全法")
        return inferred or ["ISO27001", "COBIT"]

    def _business_domain(self, text: str, topics: List[str]) -> str:
        if "财务" in text or "SOX" in text.upper():
            return "财务内控"
        if "数据" in text or "个人信息" in text:
            return "数据安全"
        if "权限" in topics or "账号" in topics:
            return "访问治理"
        return "IT治理"

    def _create_task_plan(self, audit_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        item = audit_context["audit_item"]
        standards = "、".join(audit_context["standards"])
        return [
            {"id": "T1", "name": "识别审计范围", "objective": f"确认 {item} 的边界、关键流程、系统接口和责任人。", "owner": "审计经理", "status": "done"},
            {"id": "T2", "name": "检索制度与标准", "objective": f"检索 {standards}、知识库和历史底稿中的可引用依据。", "owner": "RAG 检索器", "status": "done"},
            {"id": "T3", "name": "映射关键控制", "objective": "将风险主题映射到控制目标、控制活动和测试程序。", "owner": "控制映射器", "status": "done"},
            {"id": "T4", "name": "执行风险评分", "objective": "综合固有风险、证据质量和控制成熟度计算剩余风险。", "owner": "风险评分器", "status": "done"},
            {"id": "T5", "name": "生成审计程序", "objective": "输出抽样、访谈、系统配置检查和底稿索引。", "owner": "审计程序生成器", "status": "done"},
            {"id": "T6", "name": "质量门与复核", "objective": "判断证据充分性、缺口和是否需要人工复核。", "owner": "质量门", "status": "done"},
        ]

    def _retrieve_context(self, query: str, audit_context: Dict[str, Any]) -> Dict[str, Any]:
        rag_result: Dict[str, Any] = {"answer": "", "sources": [], "confidence": 0.0}
        if self.rag_pipeline:
            try:
                rag_result = self.rag_pipeline.query(query, audit_context)
            except Exception as exc:
                logger.warning("RAG query failed: %s", exc)
        graph = self.tools.query_knowledge_graph(audit_context["audit_item"])
        standards = self.tools.get_standards(audit_context["standards"][0] if audit_context["standards"] else None)
        return {"rag": rag_result, "graph": graph, "standards": standards}

    def _build_evidence_pack(self, retrieved: Dict[str, Any], audit_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence = []
        for index, source in enumerate(retrieved.get("rag", {}).get("sources", []), start=1):
            evidence.append(
                {
                    "id": f"RAG-{index:02d}",
                    "type": "knowledge",
                    "source": source.get("source", "knowledge"),
                    "summary": source.get("content", ""),
                    "relevance": float(source.get("score") or 0.5),
                    "usage": "作为制度依据或控制要求引用",
                }
            )
        for index, standard in enumerate(retrieved.get("standards", [])[:3], start=1):
            evidence.append(
                {
                    "id": f"STD-{index:02d}",
                    "type": "standard",
                    "source": standard.get("name", standard.get("type", "standard")),
                    "summary": standard.get("focus", standard.get("description", "")),
                    "relevance": 0.72,
                    "usage": "作为审计准则和控制设计依据",
                }
            )
        for index, edge in enumerate(retrieved.get("graph", [])[:3], start=1):
            evidence.append(
                {
                    "id": f"KG-{index:02d}",
                    "type": "graph",
                    "source": edge.get("source", "knowledge_graph"),
                    "summary": f"{edge.get('source')} -{edge.get('relation')}-> {edge.get('target')}",
                    "relevance": 0.65,
                    "usage": "用于补充系统、流程和风险关系",
                }
            )
        if not evidence:
            evidence.append(
                {
                    "id": "HEU-01",
                    "type": "heuristic",
                    "source": "内置审计控制库",
                    "summary": f"基于 {audit_context['business_domain']} 和风险主题生成初步审计假设。",
                    "relevance": 0.45,
                    "usage": "证据不足时的初步审计假设，需要现场取证验证",
                }
            )
        return evidence[:10]

    def _build_control_matrix(self, text: str, audit_context: Dict[str, Any], evidence_pack: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        standards = set(audit_context["standards"])
        query = f"{text} {audit_context['audit_item']} {' '.join(audit_context['key_risk_topics'])}"
        matrix = []
        for control in CONTROL_LIBRARY:
            keyword_hit = any(keyword.lower() in query.lower() for keyword in control["keywords"])
            standard_hit = bool(standards.intersection(control["standards"]))
            if not keyword_hit and not standard_hit:
                continue
            evidence_refs = [item["id"] for item in evidence_pack[:3]]
            maturity = 2 if keyword_hit else 3
            if len(evidence_pack) >= 4:
                maturity += 1
            matrix.append(
                {
                    "control_id": control["id"],
                    "domain": control["domain"],
                    "objective": control["objective"],
                    "test_procedure": control["test_procedure"],
                    "evidence_required": control["evidence_required"],
                    "evidence_refs": evidence_refs,
                    "maturity_level": min(maturity, 5),
                    "status": "需要取证" if evidence_pack[0]["type"] == "heuristic" else "待现场验证",
                    "risk_reduction": 0.16 if keyword_hit else 0.09,
                }
            )
        if not matrix:
            base = CONTROL_LIBRARY[0]
            matrix.append(
                {
                    "control_id": base["id"],
                    "domain": base["domain"],
                    "objective": base["objective"],
                    "test_procedure": base["test_procedure"],
                    "evidence_required": base["evidence_required"],
                    "evidence_refs": [item["id"] for item in evidence_pack[:2]],
                    "maturity_level": 2,
                    "status": "需要取证",
                    "risk_reduction": 0.08,
                }
            )
        return matrix[:6]

    def _assess_risk(
        self,
        text: str,
        audit_context: Dict[str, Any],
        retrieved: Dict[str, Any],
        control_matrix: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        matched = []
        scores = []
        normalized = text.lower()
        requested_high = any(word in normalized for word in ["高", "high", "critical", "重点关注"])
        for keyword, definition in RISK_KEYWORDS.items():
            if keyword in text or keyword in audit_context.get("audit_item", ""):
                matched.append({"topic": keyword, "domain": definition["domain"], "risk": definition["risk"], "score": definition["score"]})
                scores.append(definition["score"])
        if not matched:
            base_score = 0.76 if requested_high else 0.52
            matched.append({"topic": "通用控制", "domain": audit_context["business_domain"], "risk": "审计范围、证据链或控制责任未完全明确", "score": base_score})
            scores.append(base_score)

        control_reduction = min(sum(item["risk_reduction"] for item in control_matrix), 0.28)
        evidence_boost = 0.03 if (retrieved.get("rag") or {}).get("confidence", 0) > 0.5 else 0
        raw_score = min(sum(scores) / len(scores) + evidence_boost, 1.0)
        residual_score = max(raw_score - control_reduction / 2, 0.05)
        high = AUDIT_CONFIG["risk_threshold_high"]
        medium = AUDIT_CONFIG["risk_threshold_medium"]
        risk_level = "高" if residual_score >= high else "中" if residual_score >= medium else "低"
        return {
            "audit_item": audit_context["audit_item"],
            "inherent_risk_score": round(raw_score, 2),
            "risk_score": round(residual_score, 2),
            "risk_level": risk_level,
            "control_reduction": round(control_reduction, 2),
            "identified_risks": matched,
            "assessment_date": datetime.now().isoformat(),
        }

    def _check_compliance(self, audit_context: Dict[str, Any], risk_assessment: Dict[str, Any], control_matrix: List[Dict[str, Any]]) -> Dict[str, Any]:
        standards = audit_context["standards"]
        details = []
        avg_maturity = sum(item["maturity_level"] for item in control_matrix) / max(len(control_matrix), 1)
        for standard in standards:
            definition = AUDIT_STANDARDS.get(standard, {})
            related_controls = [
                item
                for item in control_matrix
                if standard in next((control["standards"] for control in CONTROL_LIBRARY if control["id"] == item["control_id"]), [])
            ]
            details.append(
                {
                    "standard": definition.get("name", standard),
                    "focus": definition.get("focus", "审计控制要求"),
                    "mapped_controls": [item["control_id"] for item in related_controls] or [item["control_id"] for item in control_matrix[:2]],
                    "coverage_estimate": round(min(0.95, 0.45 + len(related_controls) * 0.12 + avg_maturity * 0.06), 2),
                }
            )

        score = int(max(35, min(96, 48 + avg_maturity * 8 + len(control_matrix) * 3 - risk_assessment["risk_score"] * 18)))
        return {
            "audit_item": audit_context["audit_item"],
            "standards": standards,
            "compliance_score": score,
            "compliance_level": "高" if score >= 80 else "中" if score >= 60 else "低",
            "control_maturity_avg": round(avg_maturity, 2),
            "compliance_details": details,
            "check_date": datetime.now().isoformat(),
        }

    def _quality_gate(
        self,
        evidence_pack: List[Dict[str, Any]],
        control_matrix: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any],
        compliance_check: Dict[str, Any],
    ) -> Dict[str, Any]:
        evidence_quality = min(sum(item["relevance"] for item in evidence_pack) / max(len(evidence_pack), 1), 1.0)
        control_coverage = min(len(control_matrix) / 5, 1.0)
        completeness = 0.25 + evidence_quality * 0.35 + control_coverage * 0.25 + min(compliance_check["control_maturity_avg"] / 5, 1) * 0.15
        confidence = round(min(completeness, 0.98), 2)
        missing = []
        for control in control_matrix:
            missing.extend(control["evidence_required"][:2])
        missing = sorted(set(missing))[:8]
        status = "pass" if confidence >= 0.72 else "review" if confidence >= 0.52 else "blocked"
        return {
            "status": status,
            "confidence": confidence,
            "groundedness": round(evidence_quality, 2),
            "control_coverage": round(control_coverage, 2),
            "missing_evidence": missing,
            "escalation_required": risk_assessment["risk_level"] == "高" or status != "pass",
            "review_note": "证据较充分，可进入现场验证和复核。" if status == "pass" else "需要补充审计证据后再形成最终结论。",
        }

    def _build_audit_program(self, audit_context: Dict[str, Any], control_matrix: List[Dict[str, Any]], risk_assessment: Dict[str, Any]) -> List[Dict[str, Any]]:
        procedures = []
        for index, control in enumerate(control_matrix, start=1):
            procedures.append(
                {
                    "step_id": f"AP-{index:02d}",
                    "control_id": control["control_id"],
                    "procedure": control["test_procedure"],
                    "assertion": self._assertion_for_domain(control["domain"]),
                    "evidence": control["evidence_required"],
                    "method": "抽样检查 + 访谈确认 + 系统配置核验",
                    "priority": "高" if risk_assessment["risk_level"] == "高" else "中",
                    "workpaper_ref": f"WP-{audit_context['business_domain']}-{index:02d}",
                }
            )
        return procedures

    def _build_sampling_plan(self, audit_context: Dict[str, Any], control_matrix: List[Dict[str, Any]], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        risk_level = risk_assessment["risk_level"]
        base_size = 25 if risk_level == "高" else 15 if risk_level == "中" else 8
        return {
            "population": f"{audit_context['audit_item']} 在审计期间内的关键交易、权限、变更或日志记录",
            "period": "最近一个完整季度，必要时追溯至最近一次重大变更。",
            "method": "风险导向抽样；高风险控制采用分层抽样，关键例外全量核查。",
            "sample_size": base_size + min(len(control_matrix) * 2, 12),
            "strata": ["高权限/特权用户", "关键财务或敏感数据操作", "生产变更和例外审批", "异常日志或失败交易"],
            "exception_handling": "发现重大例外时扩大样本并触发管理层复核。",
        }

    def _draft_findings(
        self,
        audit_context: Dict[str, Any],
        control_matrix: List[Dict[str, Any]],
        risk_assessment: Dict[str, Any],
        compliance_check: Dict[str, Any],
        evidence_pack: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        findings = []
        if risk_assessment["risk_level"] in {"高", "中"}:
            primary = risk_assessment["identified_risks"][0]
            control = control_matrix[0]
            findings.append(
                {
                    "finding_id": "F-01",
                    "title": f"{audit_context['audit_item']} 存在 {primary['domain']} 控制薄弱迹象",
                    "severity": risk_assessment["risk_level"],
                    "condition": f"当前证据显示 {primary['risk']}，控制 {control['control_id']} 仍需现场验证。",
                    "criteria": "应满足最小权限、审批留痕、定期复核和异常监控等控制要求。",
                    "cause": "控制责任、证据留存或例外处理流程可能不够清晰。",
                    "effect": "可能导致越权操作、关键交易未经授权或审计追溯困难。",
                    "recommendation": f"按 {control['control_id']} 测试程序补齐证据并复核例外项。",
                    "evidence_refs": control.get("evidence_refs", []),
                }
            )
        if compliance_check["compliance_score"] < 75:
            findings.append(
                {
                    "finding_id": f"F-{len(findings) + 1:02d}",
                    "title": "合规证据覆盖不足",
                    "severity": "中",
                    "condition": f"当前合规评分为 {compliance_check['compliance_score']}，证据包数量为 {len(evidence_pack)}。",
                    "criteria": "审计结论应能追溯到制度、审批、系统配置、日志或抽样底稿。",
                    "cause": "知识库或现场证据尚未覆盖全部关键控制。",
                    "effect": "结论置信度下降，可能需要人工复核或补充取证。",
                    "recommendation": "补充制度条款、控制执行记录、抽样明细和管理层复核证据。",
                    "evidence_refs": [item["id"] for item in evidence_pack[:3]],
                }
            )
        return findings

    def _assertion_for_domain(self, domain: str) -> str:
        mapping = {
            "访问控制": "授权、完整性、职责分离",
            "身份治理": "授权、准确性、及时性",
            "职责分离": "授权、有效性",
            "变更管理": "授权、完整性、准确性",
            "业务连续性": "可用性、完整性",
            "监控审计": "完整性、可追溯性",
            "数据安全": "保密性、完整性、合规性",
            "接口安全": "完整性、准确性、可用性",
        }
        return mapping.get(domain, "完整性、授权、合规性")

    def _generate_recommendations(
        self,
        risk_assessment: Dict[str, Any],
        compliance_check: Dict[str, Any],
        control_matrix: List[Dict[str, Any]],
        quality_gate: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        recommendations = []
        priority = "高" if risk_assessment["risk_level"] == "高" else "中"
        for control in control_matrix[:4]:
            recommendations.append(
                {
                    "type": "控制整改",
                    "priority": priority,
                    "description": f"围绕 {control['domain']} 补齐控制设计和运行有效性证据。",
                    "action_items": control["evidence_required"][:3],
                    "owner_role": "控制责任人",
                    "due_days": 14 if priority == "高" else 30,
                    "success_metric": f"{control['control_id']} 关键证据齐备且抽样无重大例外",
                }
            )
        if quality_gate["missing_evidence"]:
            recommendations.append(
                {
                    "type": "证据补强",
                    "priority": "高" if quality_gate["escalation_required"] else "中",
                    "description": "补齐质量门识别的缺失证据，提升结论可审计性。",
                    "action_items": quality_gate["missing_evidence"][:4],
                    "owner_role": "审计项目经理",
                    "due_days": 7,
                    "success_metric": "质量门置信度达到 0.72 以上",
                }
            )
        recommendations.append(
            {
                "type": "持续监控",
                "priority": "低",
                "description": "将高频风险点沉淀为持续监控指标和知识库条目。",
                "action_items": ["设置关键风险指标", "按月复核异常", "更新审计知识库"],
                "owner_role": "内控与审计团队",
                "due_days": 45,
                "success_metric": "关键风险指标有监控、有阈值、有处置记录",
            }
        )
        return recommendations[:6]

    def _compose_response(
        self,
        user_input: str,
        audit_context: Dict[str, Any],
        retrieved: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        compliance_check: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        quality_gate: Dict[str, Any],
    ) -> str:
        if self.llm:
            llm_response = self._compose_with_llm(user_input, audit_context, retrieved, risk_assessment, compliance_check, recommendations, quality_gate)
            if llm_response:
                return llm_response
        standards = "、".join(compliance_check["standards"])
        top_risks = "；".join(risk["risk"] for risk in risk_assessment["identified_risks"])
        first_action = recommendations[0]["description"] if recommendations else "补齐审计证据。"
        return (
            f"审计对象：{audit_context['audit_item']}。\n\n"
            f"结论摘要：当前剩余风险等级为 {risk_assessment['risk_level']}，风险评分 {risk_assessment['risk_score']}，"
            f"控制抵减约 {risk_assessment['control_reduction']}。主要风险包括：{top_risks}。\n\n"
            f"合规视角：建议按 {standards} 取证，当前合规评分约为 {compliance_check['compliance_score']}，"
            f"控制成熟度均值 {compliance_check['control_maturity_avg']}。\n\n"
            f"质量门：状态 {quality_gate['status']}，置信度 {quality_gate['confidence']}。{quality_gate['review_note']}\n\n"
            f"优先动作：{first_action}"
        )

    def _compose_with_llm(
        self,
        user_input: str,
        audit_context: Dict[str, Any],
        retrieved: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        compliance_check: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
        quality_gate: Dict[str, Any],
    ) -> Optional[str]:
        system = (
            "你是企业智能审计 Agent。必须基于结构化事实回答，不编造证据。"
            "输出包含结论、风险、合规依据、证据缺口、整改动作和需要人工复核的点。"
        )
        payload = {
            "audit_context": audit_context,
            "retrieved": retrieved,
            "risk_assessment": risk_assessment,
            "compliance_check": compliance_check,
            "quality_gate": quality_gate,
            "recommendations": recommendations,
        }
        try:
            response = self.llm.invoke(
                [
                    SystemMessage(content=system),
                    HumanMessage(content=f"用户问题：{user_input}\n\n审计事实：{json.dumps(payload, ensure_ascii=False)}"),
                ]
            )
            return response.content
        except Exception as exc:
            logger.warning("LLM response failed, using deterministic fallback: %s", exc)
            self.llm = None
            return None

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        messages = self.session_memory.get(session_id, [])
        return [
            {
                "type": "human" if isinstance(msg, HumanMessage) else "ai",
                "content": msg.content,
                "timestamp": datetime.now().isoformat(),
            }
            for msg in messages
        ]

    def get_service_status(self) -> Dict[str, bool]:
        return {"llm": bool(self.llm), "mysql": self.tools.status.mysql, "neo4j": self.tools.status.neo4j, "rag": bool(self.rag_pipeline)}

    def close(self) -> None:
        self.tools.close()

"""
Audit agent orchestration.

The agent follows a pragmatic planner -> retrieval -> risk/compliance ->
response workflow. External systems are optional: when the LLM, MySQL, Neo4j or
RAG store are unavailable, deterministic audit heuristics keep the app usable.
"""

from __future__ import annotations

import json
import logging
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
        "focus": "IT 治理、价值交付、风险优化和资源优化",
        "controls": ["治理目标映射", "流程责任矩阵", "绩效指标", "风险场景管理"],
    },
    "ISO27001": {
        "name": "ISO/IEC 27001",
        "focus": "信息安全管理体系、风险评估和控制措施",
        "controls": ["访问控制", "资产管理", "事件响应", "供应商安全", "备份与恢复"],
    },
    "SOX": {
        "name": "Sarbanes-Oxley Act",
        "focus": "财务报告内部控制、变更审批、职责分离和审计证据",
        "controls": ["职责分离", "变更管理", "日志留存", "财务数据完整性", "管理层复核"],
    },
    "数据安全法": {
        "name": "数据安全法",
        "focus": "数据分类分级、重要数据保护、风险监测和应急处置",
        "controls": ["分类分级", "最小权限", "数据加密", "安全评估", "应急预案"],
    },
}

RISK_KEYWORDS: Dict[str, Dict[str, Any]] = {
    "权限": {"score": 0.82, "risk": "权限滥用或职责分离不足", "controls": ["最小权限", "定期复核", "强制审批"]},
    "账号": {"score": 0.78, "risk": "账号生命周期管理不足", "controls": ["入离转调流程", "多因素认证", "异常登录监控"]},
    "财务": {"score": 0.86, "risk": "财务数据完整性和审批链路风险", "controls": ["凭证校验", "职责分离", "管理层复核"]},
    "变更": {"score": 0.74, "risk": "系统变更未经充分测试或审批", "controls": ["变更委员会", "回退方案", "上线后复核"]},
    "备份": {"score": 0.68, "risk": "备份不可用或恢复目标不清晰", "controls": ["恢复演练", "异地备份", "RPO/RTO 定义"]},
    "日志": {"score": 0.65, "risk": "审计日志不完整或不可追溯", "controls": ["集中日志", "防篡改存储", "告警规则"]},
    "数据": {"score": 0.76, "risk": "敏感数据泄露或使用不合规", "controls": ["分类分级", "脱敏", "加密", "访问审计"]},
    "接口": {"score": 0.7, "risk": "接口调用缺少鉴权、限流或监控", "controls": ["鉴权签名", "速率限制", "接口台账"]},
}


@dataclass
class ServiceStatus:
    llm: bool = False
    mysql: bool = False
    neo4j: bool = False
    rag: bool = False


class OptionalAuditTools:
    """Connects to external audit data sources only when they are available."""

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
            logger.info("MySQL unavailable, falling back to built-in standards: %s", exc)

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
            logger.info("Neo4j unavailable, falling back to heuristic graph lookup: %s", exc)
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
    """High-level agent used by the FastAPI layer."""

    def __init__(self, rag_pipeline: Any = None) -> None:
        self.tools = OptionalAuditTools()
        self.rag_pipeline = rag_pipeline
        self.session_memory: Dict[str, List[BaseMessage]] = {}
        self.llm = self._init_llm()
        logger.info("AuditAgent initialized. LLM=%s MySQL=%s Neo4j=%s", bool(self.llm), self.tools.status.mysql, self.tools.status.neo4j)

    def _init_llm(self) -> Any:
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

        audit_context = self._extract_context(user_input)
        retrieved = self._retrieve_context(user_input, audit_context)
        risk_assessment = self._assess_risk(user_input, audit_context, retrieved)
        compliance_check = self._check_compliance(audit_context, risk_assessment)
        recommendations = self._generate_recommendations(risk_assessment, compliance_check)
        response = self._compose_response(user_input, audit_context, retrieved, risk_assessment, compliance_check, recommendations)

        self.session_memory[session_id].append(AIMessage(content=response))
        self._trim_session(session_id)

        return {
            "session_id": session_id,
            "response": response,
            "audit_context": audit_context,
            "retrieval": retrieved,
            "risk_assessment": risk_assessment,
            "compliance_check": compliance_check,
            "recommendations": recommendations,
            "service_status": self.get_service_status(),
        }

    def _trim_session(self, session_id: str) -> None:
        max_messages = AUDIT_CONFIG["max_session_messages"]
        self.session_memory[session_id] = self.session_memory[session_id][-max_messages:]

    def _extract_context(self, text: str) -> Dict[str, Any]:
        standards = [key for key in AUDIT_STANDARDS if key.lower() in text.lower()]
        if "ISO" in text.upper() and "ISO27001" not in standards:
            standards.append("ISO27001")

        audit_types = []
        for keyword in ["安全审计", "合规审计", "风险评估", "内部控制审计", "数据审计", "财务审计"]:
            if keyword in text:
                audit_types.append(keyword)

        item = self._guess_audit_item(text)
        return {
            "audit_item": item,
            "audit_types": audit_types or ["综合审计分析"],
            "standards": standards or self._infer_standards(text),
            "key_risk_topics": [key for key in RISK_KEYWORDS if key in text],
            "generated_at": datetime.now().isoformat(),
        }

    def _guess_audit_item(self, text: str) -> str:
        patterns = [
            r"对(.+?)进行",
            r"评估(.+?)的",
            r"检查(.+?)是否",
            r"分析(.+?)的",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1).strip(" ，。")
        for keyword in ["ERP系统", "CRM系统", "财务系统", "权限管理", "数据备份", "日志管理", "接口管理"]:
            if keyword in text:
                return keyword
        return "待审计对象"

    def _infer_standards(self, text: str) -> List[str]:
        inferred = []
        if any(word in text for word in ["财务", "报表", "凭证", "SOX"]):
            inferred.append("SOX")
        if any(word in text for word in ["安全", "权限", "账号", "日志", "备份"]):
            inferred.append("ISO27001")
        if any(word in text for word in ["治理", "IT", "系统", "流程"]):
            inferred.append("COBIT")
        if any(word in text for word in ["数据", "个人信息", "敏感"]):
            inferred.append("数据安全法")
        return inferred or ["ISO27001", "COBIT"]

    def _retrieve_context(self, question: str, audit_context: Dict[str, Any]) -> Dict[str, Any]:
        kg_results = self.tools.query_knowledge_graph(audit_context["audit_item"])
        rag_result = None
        if self.rag_pipeline:
            try:
                rag_result = self.rag_pipeline.query(question, audit_context, k=5)
            except Exception as exc:
                logger.warning("RAG retrieval failed: %s", exc)

        standards = []
        for standard in audit_context["standards"]:
            standards.extend(self.tools.get_standards(standard))

        return {
            "knowledge_graph": kg_results,
            "rag": rag_result,
            "standards": standards,
        }

    def _assess_risk(self, text: str, audit_context: Dict[str, Any], retrieved: Dict[str, Any]) -> Dict[str, Any]:
        matched = []
        scores = []
        for keyword, definition in RISK_KEYWORDS.items():
            if keyword in text or keyword in audit_context.get("audit_item", ""):
                matched.append(
                    {
                        "topic": keyword,
                        "risk": definition["risk"],
                        "score": definition["score"],
                        "controls": definition["controls"],
                    }
                )
                scores.append(definition["score"])

        if not matched:
            matched.append(
                {
                    "topic": "通用控制",
                    "risk": "审计范围、证据链或控制责任未完全明确",
                    "score": 0.52,
                    "controls": ["明确控制责任人", "补齐审计证据", "建立例行复核"],
                }
            )
            scores.append(0.52)

        if retrieved.get("knowledge_graph"):
            scores.append(0.05 + max(scores))
        rag_result = retrieved.get("rag") or {}
        if rag_result.get("confidence", 0) > 0.65:
            scores.append(min(0.95, max(scores) + 0.04))

        risk_score = min(sum(scores) / len(scores), 1.0)
        high = AUDIT_CONFIG["risk_threshold_high"]
        medium = AUDIT_CONFIG["risk_threshold_medium"]
        risk_level = "高" if risk_score >= high else "中" if risk_score >= medium else "低"

        return {
            "audit_item": audit_context["audit_item"],
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "identified_risks": matched,
            "assessment_date": datetime.now().isoformat(),
        }

    def _check_compliance(self, audit_context: Dict[str, Any], risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        standards = audit_context["standards"]
        details = []
        for standard in standards:
            definition = AUDIT_STANDARDS.get(standard, {})
            controls = definition.get("controls", [])
            coverage = max(0.45, 1.0 - risk_assessment["risk_score"] / 2)
            details.append(
                {
                    "standard": definition.get("name", standard),
                    "focus": definition.get("focus", "审计控制要求"),
                    "suggested_controls": controls,
                    "coverage_estimate": round(coverage, 2),
                }
            )

        score = int(max(35, min(95, 100 - risk_assessment["risk_score"] * 45 + len(details) * 3)))
        return {
            "audit_item": audit_context["audit_item"],
            "standards": standards,
            "compliance_score": score,
            "compliance_level": "高" if score >= 80 else "中" if score >= 60 else "低",
            "compliance_details": details,
            "check_date": datetime.now().isoformat(),
        }

    def _generate_recommendations(self, risk_assessment: Dict[str, Any], compliance_check: Dict[str, Any]) -> List[Dict[str, Any]]:
        recommendations = []
        priority = "高" if risk_assessment["risk_level"] == "高" else "中"
        for risk in risk_assessment["identified_risks"]:
            recommendations.append(
                {
                    "type": "风险控制",
                    "priority": priority,
                    "description": f"针对“{risk['risk']}”建立可验证的控制闭环。",
                    "action_items": risk["controls"],
                }
            )

        if compliance_check["compliance_score"] < 80:
            recommendations.append(
                {
                    "type": "合规整改",
                    "priority": "中",
                    "description": "补齐控制证据、审批记录和例外处理说明，形成可审计轨迹。",
                    "action_items": ["建立证据清单", "明确责任人和整改期限", "复核关键控制执行记录"],
                }
            )

        recommendations.append(
            {
                "type": "持续监控",
                "priority": "低",
                "description": "将高频风险点纳入周期性监控和复盘。",
                "action_items": ["设置关键风险指标", "按月复核异常", "沉淀审计知识库"],
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
    ) -> str:
        if self.llm:
            llm_response = self._compose_with_llm(user_input, audit_context, retrieved, risk_assessment, compliance_check, recommendations)
            if llm_response:
                return llm_response

        standards = "、".join(compliance_check["standards"])
        top_risks = "；".join(risk["risk"] for risk in risk_assessment["identified_risks"])
        actions = "；".join(recommendations[0]["action_items"]) if recommendations else "补齐审计证据"
        return (
            f"审计对象：{audit_context['audit_item']}。\n\n"
            f"初步判断：当前风险等级为{risk_assessment['risk_level']}，风险评分 {risk_assessment['risk_score']}。"
            f"主要风险包括：{top_risks}。\n\n"
            f"合规视角：建议按 {standards} 对控制设计和运行有效性取证，当前合规评分约为 "
            f"{compliance_check['compliance_score']}，等级为{compliance_check['compliance_level']}。\n\n"
            f"优先整改：{actions}。同时保留审批、日志、抽样记录和复核结论，便于形成完整审计证据链。"
        )

    def _compose_with_llm(
        self,
        user_input: str,
        audit_context: Dict[str, Any],
        retrieved: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        compliance_check: Dict[str, Any],
        recommendations: List[Dict[str, Any]],
    ) -> Optional[str]:
        system = (
            "你是企业智能审计 Agent。请基于给定结构化事实回答，不要编造不存在的制度条款。"
            "输出应包含：结论、风险、合规依据、整改动作和需要补充的证据。"
        )
        payload = {
            "audit_context": audit_context,
            "retrieved": retrieved,
            "risk_assessment": risk_assessment,
            "compliance_check": compliance_check,
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
        history = []
        for msg in messages:
            history.append(
                {
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat(),
                }
            )
        return history

    def get_service_status(self) -> Dict[str, bool]:
        return {
            "llm": bool(self.llm),
            "mysql": self.tools.status.mysql,
            "neo4j": self.tools.status.neo4j,
            "rag": bool(self.rag_pipeline),
        }

    def close(self) -> None:
        self.tools.close()

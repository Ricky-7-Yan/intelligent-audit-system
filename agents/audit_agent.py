"""
基于LangGraph的智能审计Agent
Intelligent Audit Agent based on LangGraph
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import Tool
import json
import logging
from datetime import datetime
from config import LLM_CONFIG, NEO4J_CONFIG, MYSQL_CONFIG
import pymysql
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AuditState(TypedDict):
    """审计状态定义"""
    messages: Annotated[List[BaseMessage], "对话消息列表"]
    current_audit_item: Optional[Dict[str, Any]]
    audit_context: Dict[str, Any]
    risk_assessment: Optional[Dict[str, Any]]
    compliance_check: Optional[Dict[str, Any]]
    recommendations: List[Dict[str, Any]]
    next_action: str
    user_input: str
    session_id: str

class AuditTools:
    """审计工具集"""

    def __init__(self):
        self.neo4j_driver = GraphDatabase.driver(
            NEO4J_CONFIG['uri'],
            auth=(NEO4J_CONFIG['user'], NEO4J_CONFIG['password'])
        )
        self.mysql_connection = pymysql.connect(**MYSQL_CONFIG)

    def close(self):
        self.neo4j_driver.close()
        self.mysql_connection.close()

    def query_knowledge_graph(self, query: str) -> str:
        """查询知识图谱"""
        try:
            with self.neo4j_driver.session() as session:
                # 构建Cypher查询
                cypher_query = """
                MATCH (n)-[r]->(m)
                WHERE n.name CONTAINS $query OR m.name CONTAINS $query 
                   OR n.description CONTAINS $query OR m.description CONTAINS $query
                RETURN n.name as source_name, n.type as source_type, 
                       type(r) as relation_type, 
                       m.name as target_name, m.type as target_type,
                       n.description as source_desc, m.description as target_desc
                LIMIT 20
                """

                result = session.run(cypher_query, query=query)
                results = []

                for record in result:
                    results.append({
                        'source': f"{record['source_name']} ({record['source_type']})",
                        'relation': record['relation_type'],
                        'target': f"{record['target_name']} ({record['target_type']})",
                        'source_desc': record['source_desc'],
                        'target_desc': record['target_desc']
                    })

                if results:
                    return json.dumps(results, ensure_ascii=False, indent=2)
                else:
                    return f"未找到与'{query}'相关的知识图谱信息"

        except Exception as e:
            logger.error(f"知识图谱查询失败: {e}")
            return f"知识图谱查询失败: {str(e)}"

    def get_audit_standards(self, standard_type: str = None) -> str:
        """获取审计标准"""
        try:
            with self.mysql_connection.cursor() as cursor:
                if standard_type:
                    cursor.execute("""
                        SELECT standard_name, standard_type, version, description, requirements
                        FROM audit_standards 
                        WHERE standard_type = %s
                    """, (standard_type,))
                else:
                    cursor.execute("""
                        SELECT standard_name, standard_type, version, description, requirements
                        FROM audit_standards
                    """)

                results = cursor.fetchall()
                standards = []

                for row in results:
                    standards.append({
                        'name': row[0],
                        'type': row[1],
                        'version': row[2],
                        'description': row[3],
                        'requirements': json.loads(row[4]) if row[4] else {}
                    })

                return json.dumps(standards, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"获取审计标准失败: {e}")
            return f"获取审计标准失败: {str(e)}"

    def assess_risk(self, audit_item: str, context: Dict[str, Any] = None) -> str:
        """风险评估"""
        try:
            # 从知识图谱中查找相关风险
            with self.neo4j_driver.session() as session:
                risk_query = """
                MATCH (p:Process {name: $item_name})-[:MITIGATES]->(r:Risk)
                RETURN r.name as risk_name, r.level as risk_level, r.description as risk_desc
                UNION
                MATCH (s:System {name: $item_name})-[:HAS_RISK]->(r:Risk)
                RETURN r.name as risk_name, r.level as risk_level, r.description as risk_desc
                """

                result = session.run(risk_query, item_name=audit_item)
                risks = []

                for record in result:
                    risks.append({
                        'name': record['risk_name'],
                        'level': record['risk_level'],
                        'description': record['risk_desc']
                    })

                # 计算风险评分
                risk_score = 0
                risk_factors = {
                    '高': 0.9,
                    '中': 0.6,
                    '低': 0.3
                }

                for risk in risks:
                    risk_score += risk_factors.get(risk['level'], 0.5)

                if risks:
                    risk_score = min(risk_score / len(risks), 1.0)
                else:
                    risk_score = 0.5  # 默认中等风险

                assessment = {
                    'audit_item': audit_item,
                    'risk_score': round(risk_score, 2),
                    'risk_level': '高' if risk_score > 0.7 else '中' if risk_score > 0.4 else '低',
                    'identified_risks': risks,
                    'assessment_date': datetime.now().isoformat()
                }

                return json.dumps(assessment, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"风险评估失败: {e}")
            return f"风险评估失败: {str(e)}"

    def check_compliance(self, audit_item: str, standard_type: str) -> str:
        """合规性检查"""
        try:
            # 查询相关标准和流程
            with self.neo4j_driver.session() as session:
                compliance_query = """
                MATCH (s:Standard {type: $standard_type})-[:COVERS]->(p:Process {name: $item_name})
                RETURN s.name as standard_name, s.description as standard_desc,
                       p.name as process_name, p.description as process_desc
                """

                result = session.run(compliance_query,
                                  standard_type=standard_type,
                                  item_name=audit_item)

                compliance_results = []
                for record in result:
                    compliance_results.append({
                        'standard': record['standard_name'],
                        'standard_desc': record['standard_desc'],
                        'process': record['process_name'],
                        'process_desc': record['process_desc'],
                        'compliance_status': '符合'  # 简化处理
                    })

                # 计算合规性评分
                compliance_score = len(compliance_results) * 20  # 每个符合项20分
                compliance_score = min(compliance_score, 100)

                compliance_check = {
                    'audit_item': audit_item,
                    'standard_type': standard_type,
                    'compliance_score': compliance_score,
                    'compliance_level': '高' if compliance_score >= 80 else '中' if compliance_score >= 60 else '低',
                    'compliance_details': compliance_results,
                    'check_date': datetime.now().isoformat()
                }

                return json.dumps(compliance_check, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"合规性检查失败: {e}")
            return f"合规性检查失败: {str(e)}"

    def generate_recommendations(self, audit_item: str, risk_assessment: Dict[str, Any],
                                compliance_check: Dict[str, Any]) -> str:
        """生成整改建议"""
        try:
            recommendations = []

            # 基于风险评估生成建议
            if risk_assessment.get('risk_score', 0) > 0.7:
                recommendations.append({
                    'type': '风险控制',
                    'priority': '高',
                    'description': '建议加强风险控制措施，实施更严格的安全防护',
                    'action_items': [
                        '加强访问控制',
                        '实施数据加密',
                        '建立监控机制'
                    ]
                })

            # 基于合规性检查生成建议
            if compliance_check.get('compliance_score', 0) < 80:
                recommendations.append({
                    'type': '合规改进',
                    'priority': '中',
                    'description': '建议改进合规性措施，确保符合相关标准',
                    'action_items': [
                        '更新流程文档',
                        '加强员工培训',
                        '建立合规检查机制'
                    ]
                })

            # 通用建议
            recommendations.append({
                'type': '持续改进',
                'priority': '低',
                'description': '建议建立持续监控和改进机制',
                'action_items': [
                    '定期风险评估',
                    '持续合规检查',
                    '员工安全意识培训'
                ]
            })

            return json.dumps(recommendations, ensure_ascii=False, indent=2)

        except Exception as e:
            logger.error(f"生成建议失败: {e}")
            return f"生成建议失败: {str(e)}"

class AuditAgent:
    """智能审计Agent - 简化版本，不依赖LangGraph"""

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=LLM_CONFIG['api_key'],
            base_url=LLM_CONFIG['base_url'],
            model=LLM_CONFIG['model'],
            temperature=LLM_CONFIG['temperature'],
            max_tokens=LLM_CONFIG['max_tokens']
        )

        self.tools = AuditTools()
        self.session_memory = {}

        # 创建工具
        self.audit_tools = [
            Tool(
                name="query_knowledge_graph",
                description="查询知识图谱，获取审计相关的实体和关系信息",
                func=self.tools.query_knowledge_graph
            ),
            Tool(
                name="get_audit_standards",
                description="获取审计标准信息，包括COBIT、ISO27001、SOX等",
                func=self.tools.get_audit_standards
            ),
            Tool(
                name="assess_risk",
                description="对审计项目进行风险评估",
                func=self.tools.assess_risk
            ),
            Tool(
                name="check_compliance",
                description="检查审计项目的合规性",
                func=self.tools.check_compliance
            ),
            Tool(
                name="generate_recommendations",
                description="基于审计结果生成整改建议",
                func=self.tools.generate_recommendations
            )
        ]

        # 创建提示模板
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的审计专家AI助手，具有以下能力：
1. 深度理解审计标准和法规（COBIT、ISO27001、SOX、数据安全法等）
2. 进行风险评估和合规性检查
3. 基于知识图谱提供专业的审计建议
4. 支持多轮对话式审计决策

请根据用户的问题和上下文，使用可用的工具进行分析，并提供专业、准确的审计建议。
回答要具体、可操作，并引用相关的标准和法规。"""),
            ("human", "{input}")
        ])

        logger.info("智能审计Agent初始化完成")

    def process_audit_query(self, user_input: str, session_id: str = None) -> Dict[str, Any]:
        """处理审计查询"""
        try:
            if not session_id:
                session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            # 获取会话历史
            if session_id not in self.session_memory:
                self.session_memory[session_id] = []

            # 添加用户输入到历史
            self.session_memory[session_id].append(HumanMessage(content=user_input))

            # 构建上下文
            messages = self.session_memory[session_id][-5:]  # 只保留最近5条消息

            # 调用LLM
            response = self.llm.invoke(messages)

            # 添加AI回复到历史
            self.session_memory[session_id].append(AIMessage(content=response.content))

            # 解析响应，提取审计信息
            audit_context = self._parse_audit_response(response.content)

            return {
                "session_id": session_id,
                "response": response.content,
                "audit_context": audit_context,
                "risk_assessment": None,
                "compliance_check": None,
                "recommendations": []
            }

        except Exception as e:
            logger.error(f"处理审计查询失败: {e}")
            return {
                "session_id": session_id,
                "response": f"抱歉，处理您的请求时出现错误: {str(e)}",
                "audit_context": {},
                "risk_assessment": None,
                "compliance_check": None,
                "recommendations": []
            }

    def _parse_audit_response(self, response: str) -> Dict[str, Any]:
        """解析AI响应，提取审计信息"""
        try:
            # 尝试从响应中提取JSON信息
            import re
            json_pattern = r'\{.*\}'
            json_matches = re.findall(json_pattern, response, re.DOTALL)

            audit_info = {}
            for match in json_matches:
                try:
                    data = json.loads(match)
                    audit_info.update(data)
                except:
                    continue

            return audit_info

        except Exception as e:
            logger.error(f"解析审计响应失败: {e}")
            return {}

    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """获取会话历史"""
        try:
            if session_id not in self.session_memory:
                return []

            messages = self.session_memory[session_id]
            history = []

            for msg in messages:
                history.append({
                    "type": "human" if isinstance(msg, HumanMessage) else "ai",
                    "content": msg.content,
                    "timestamp": datetime.now().isoformat()
                })

            return history

        except Exception as e:
            logger.error(f"获取会话历史失败: {e}")
            return []

    def close(self):
        """关闭Agent"""
        try:
            self.tools.close()
        except:
            pass

# 使用示例
if __name__ == "__main__":
    agent = AuditAgent()

    try:
        # 测试审计查询
        test_queries = [
            "请对ERP系统进行安全审计",
            "评估用户权限管理的风险",
            "检查财务报告流程是否符合SOX要求",
            "分析数据备份流程的合规性"
        ]

        for query in test_queries:
            print(f"\n查询: {query}")
            result = agent.process_audit_query(query)
            print(f"响应: {result['response']}")
            print(f"审计上下文: {result['audit_context']}")

    except Exception as e:
        logger.error(f"Agent测试失败: {e}")
    finally:
        agent.close()

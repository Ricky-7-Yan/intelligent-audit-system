"""
Neo4j知识图谱初始化脚本
Neo4j Knowledge Graph Initialization Script
"""

from neo4j import GraphDatabase
from config import NEO4J_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Neo4jManager:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_CONFIG['uri'],
            auth=(NEO4J_CONFIG['user'], NEO4J_CONFIG['password'])
        )

    def close(self):
        self.driver.close()

    def create_constraints(self):
        """创建约束和索引"""
        with self.driver.session() as session:
            # 创建唯一性约束
            constraints = [
                "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
                "CREATE CONSTRAINT standard_id_unique IF NOT EXISTS FOR (s:Standard) REQUIRE s.id IS UNIQUE",
                "CREATE CONSTRAINT process_id_unique IF NOT EXISTS FOR (p:Process) REQUIRE p.id IS UNIQUE",
                "CREATE CONSTRAINT risk_id_unique IF NOT EXISTS FOR (r:Risk) REQUIRE r.id IS UNIQUE",
                "CREATE CONSTRAINT control_id_unique IF NOT EXISTS FOR (c:Control) REQUIRE c.id IS UNIQUE"
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"约束创建成功: {constraint}")
                except Exception as e:
                    logger.warning(f"约束创建失败（可能已存在）: {e}")

    def create_indexes(self):
        """创建索引"""
        with self.driver.session() as session:
            indexes = [
                "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
                "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
                "CREATE INDEX standard_name_index IF NOT EXISTS FOR (s:Standard) ON (s.name)",
                "CREATE INDEX process_name_index IF NOT EXISTS FOR (p:Process) ON (p.name)",
                "CREATE INDEX risk_level_index IF NOT EXISTS FOR (r:Risk) ON (r.level)"
            ]

            for index in indexes:
                try:
                    session.run(index)
                    logger.info(f"索引创建成功: {index}")
                except Exception as e:
                    logger.warning(f"索引创建失败（可能已存在）: {e}")

    def create_initial_nodes(self):
        """创建初始节点"""
        with self.driver.session() as session:
            # 创建审计标准节点
            standards = [
                {
                    'id': 'cobit_2019',
                    'name': 'COBIT 2019',
                    'type': 'COBIT',
                    'version': '2019',
                    'description': 'COBIT 2019框架为IT治理和管理提供全面的指导',
                    'domains': ['治理', '管理'],
                    'processes': 40,
                    'principles': 5
                },
                {
                    'id': 'iso27001_2022',
                    'name': 'ISO/IEC 27001:2022',
                    'type': 'ISO27001',
                    'version': '2022',
                    'description': '信息安全管理体系国际标准',
                    'controls': 93,
                    'categories': 4,
                    'annexes': 14
                },
                {
                    'id': 'sox_2002',
                    'name': 'SOX法案',
                    'type': 'SOX',
                    'version': '2002',
                    'description': '萨班斯-奥克斯利法案，规范上市公司财务报告',
                    'sections': 11,
                    'requirements': ['内部控制', '财务报告', '审计委员会']
                },
                {
                    'id': 'data_security_law',
                    'name': '数据安全法',
                    'type': '数据安全法',
                    'version': '2021',
                    'description': '中华人民共和国数据安全法',
                    'chapters': 7,
                    'articles': 55,
                    'focus': ['数据分类', '数据保护', '数据跨境']
                },
                {
                    'id': 'cybersecurity_law',
                    'name': '网络安全法',
                    'type': '网络安全法',
                    'version': '2017',
                    'description': '中华人民共和国网络安全法',
                    'chapters': 7,
                    'articles': 79,
                    'focus': ['网络运行安全', '网络信息安全', '监测预警']
                }
            ]

            for standard in standards:
                # 为每个标准分别处理，只设置存在的字段
                query = """
                    MERGE (s:Standard {id: $id})
                    SET s.name = $name,
                        s.type = $type,
                        s.version = $version,
                        s.description = $description
                """

                # 添加可选字段
                if 'domains' in standard:
                    query += ", s.domains = $domains"
                if 'processes' in standard:
                    query += ", s.processes = $processes"
                if 'principles' in standard:
                    query += ", s.principles = $principles"
                if 'controls' in standard:
                    query += ", s.controls = $controls"
                if 'categories' in standard:
                    query += ", s.categories = $categories"
                if 'annexes' in standard:
                    query += ", s.annexes = $annexes"
                if 'sections' in standard:
                    query += ", s.sections = $sections"
                if 'requirements' in standard:
                    query += ", s.requirements = $requirements"
                if 'chapters' in standard:
                    query += ", s.chapters = $chapters"
                if 'articles' in standard:
                    query += ", s.articles = $articles"
                if 'focus' in standard:
                    query += ", s.focus = $focus"

                session.run(query, **standard)

            # 创建业务流程节点
            processes = [
                {
                    'id': 'proc_001',
                    'name': '用户权限管理',
                    'description': '管理用户账户的创建、修改、删除和权限分配',
                    'risk_level': '中',
                    'category': 'IT管理'
                },
                {
                    'id': 'proc_002',
                    'name': '财务报告流程',
                    'description': '财务数据的收集、处理、审核和报告生成',
                    'risk_level': '高',
                    'category': '财务管理'
                },
                {
                    'id': 'proc_003',
                    'name': '数据备份与恢复',
                    'description': '定期备份重要数据并建立恢复机制',
                    'risk_level': '中',
                    'category': 'IT运维'
                },
                {
                    'id': 'proc_004',
                    'name': '变更管理',
                    'description': 'IT系统和应用程序的变更控制流程',
                    'risk_level': '高',
                    'category': 'IT管理'
                },
                {
                    'id': 'proc_005',
                    'name': '事件响应',
                    'description': '安全事件的检测、分析和响应流程',
                    'risk_level': '高',
                    'category': '安全管理'
                }
            ]

            for process in processes:
                session.run("""
                    MERGE (p:Process {id: $id})
                    SET p.name = $name,
                        p.description = $description,
                        p.risk_level = $risk_level,
                        p.category = $category
                """, **process)

            # 创建风险节点
            risks = [
                {
                    'id': 'risk_001',
                    'name': '数据泄露风险',
                    'description': '敏感数据被未授权访问或泄露的风险',
                    'level': '高',
                    'category': '信息安全'
                },
                {
                    'id': 'risk_002',
                    'name': '系统可用性风险',
                    'description': '关键系统不可用导致业务中断的风险',
                    'level': '高',
                    'category': '运营风险'
                },
                {
                    'id': 'risk_003',
                    'name': '合规性风险',
                    'description': '违反相关法律法规和行业标准的风险',
                    'level': '中',
                    'category': '合规风险'
                },
                {
                    'id': 'risk_004',
                    'name': '财务报告风险',
                    'description': '财务报告不准确或存在重大错报的风险',
                    'level': '高',
                    'category': '财务风险'
                },
                {
                    'id': 'risk_005',
                    'name': '第三方风险',
                    'description': '第三方供应商或合作伙伴带来的风险',
                    'level': '中',
                    'category': '供应链风险'
                }
            ]

            for risk in risks:
                session.run("""
                    MERGE (r:Risk {id: $id})
                    SET r.name = $name,
                        r.description = $description,
                        r.level = $level,
                        r.category = $category
                """, **risk)

            # 创建控制措施节点
            controls = [
                {
                    'id': 'ctrl_001',
                    'name': '访问控制',
                    'description': '通过身份认证和授权机制控制用户访问',
                    'type': '预防性控制',
                    'effectiveness': '高'
                },
                {
                    'id': 'ctrl_002',
                    'name': '数据加密',
                    'description': '对敏感数据进行加密保护',
                    'type': '预防性控制',
                    'effectiveness': '高'
                },
                {
                    'id': 'ctrl_003',
                    'name': '审计日志',
                    'description': '记录系统操作和用户行为日志',
                    'type': '检测性控制',
                    'effectiveness': '中'
                },
                {
                    'id': 'ctrl_004',
                    'name': '备份恢复',
                    'description': '定期备份数据并建立恢复机制',
                    'type': '纠正性控制',
                    'effectiveness': '高'
                },
                {
                    'id': 'ctrl_005',
                    'name': '变更审批',
                    'description': '对系统变更进行审批和测试',
                    'type': '预防性控制',
                    'effectiveness': '中'
                }
            ]

            for control in controls:
                session.run("""
                    MERGE (c:Control {id: $id})
                    SET c.name = $name,
                        c.description = $description,
                        c.type = $type,
                        c.effectiveness = $effectiveness
                """, **control)

            logger.info("初始节点创建完成")

    def create_initial_relationships(self):
        """创建初始关系"""
        with self.driver.session() as session:
            # 标准与流程的关系
            standard_process_relations = [
                ('cobit_2019', 'proc_001', 'COVERS', {'coverage': 0.8}),
                ('cobit_2019', 'proc_002', 'COVERS', {'coverage': 0.9}),
                ('cobit_2019', 'proc_003', 'COVERS', {'coverage': 0.7}),
                ('cobit_2019', 'proc_004', 'COVERS', {'coverage': 0.85}),
                ('iso27001_2022', 'proc_001', 'COVERS', {'coverage': 0.9}),
                ('iso27001_2022', 'proc_003', 'COVERS', {'coverage': 0.8}),
                ('iso27001_2022', 'proc_005', 'COVERS', {'coverage': 0.95}),
                ('sox_2002', 'proc_002', 'COVERS', {'coverage': 0.95}),
                ('data_security_law', 'proc_001', 'COVERS', {'coverage': 0.7}),
                ('data_security_law', 'proc_003', 'COVERS', {'coverage': 0.8})
            ]

            for std_id, proc_id, rel_type, properties in standard_process_relations:
                session.run("""
                    MATCH (s:Standard {id: $std_id})
                    MATCH (p:Process {id: $proc_id})
                    MERGE (s)-[r:COVERS]->(p)
                    SET r.coverage = $coverage
                """, std_id=std_id, proc_id=proc_id, coverage=properties['coverage'])

            # 流程与风险的关系
            process_risk_relations = [
                ('proc_001', 'risk_001', 'MITIGATES', {'mitigation_level': 0.8}),
                ('proc_001', 'risk_003', 'MITIGATES', {'mitigation_level': 0.7}),
                ('proc_002', 'risk_004', 'MITIGATES', {'mitigation_level': 0.9}),
                ('proc_003', 'risk_002', 'MITIGATES', {'mitigation_level': 0.85}),
                ('proc_004', 'risk_002', 'MITIGATES', {'mitigation_level': 0.7}),
                ('proc_005', 'risk_001', 'MITIGATES', {'mitigation_level': 0.9}),
                ('proc_005', 'risk_002', 'MITIGATES', {'mitigation_level': 0.8})
            ]

            for proc_id, risk_id, rel_type, properties in process_risk_relations:
                session.run("""
                    MATCH (p:Process {id: $proc_id})
                    MATCH (r:Risk {id: $risk_id})
                    MERGE (p)-[rel:MITIGATES]->(r)
                    SET rel.mitigation_level = $mitigation_level
                """, proc_id=proc_id, risk_id=risk_id, mitigation_level=properties['mitigation_level'])

            # 控制措施与风险的关系
            control_risk_relations = [
                ('ctrl_001', 'risk_001', 'ADDRESSES', {'effectiveness': 0.9}),
                ('ctrl_002', 'risk_001', 'ADDRESSES', {'effectiveness': 0.95}),
                ('ctrl_003', 'risk_001', 'ADDRESSES', {'effectiveness': 0.7}),
                ('ctrl_003', 'risk_004', 'ADDRESSES', {'effectiveness': 0.8}),
                ('ctrl_004', 'risk_002', 'ADDRESSES', {'effectiveness': 0.9}),
                ('ctrl_005', 'risk_002', 'ADDRESSES', {'effectiveness': 0.8}),
                ('ctrl_005', 'risk_003', 'ADDRESSES', {'effectiveness': 0.7})
            ]

            for ctrl_id, risk_id, rel_type, properties in control_risk_relations:
                session.run("""
                    MATCH (c:Control {id: $ctrl_id})
                    MATCH (r:Risk {id: $risk_id})
                    MERGE (c)-[rel:ADDRESSES]->(r)
                    SET rel.effectiveness = $effectiveness
                """, ctrl_id=ctrl_id, risk_id=risk_id, effectiveness=properties['effectiveness'])

            # 控制措施与流程的关系
            control_process_relations = [
                ('ctrl_001', 'proc_001', 'IMPLEMENTS', {'implementation_level': 0.9}),
                ('ctrl_002', 'proc_003', 'IMPLEMENTS', {'implementation_level': 0.8}),
                ('ctrl_003', 'proc_001', 'IMPLEMENTS', {'implementation_level': 0.7}),
                ('ctrl_003', 'proc_002', 'IMPLEMENTS', {'implementation_level': 0.8}),
                ('ctrl_004', 'proc_003', 'IMPLEMENTS', {'implementation_level': 0.9}),
                ('ctrl_005', 'proc_004', 'IMPLEMENTS', {'implementation_level': 0.8})
            ]

            for ctrl_id, proc_id, rel_type, properties in control_process_relations:
                session.run("""
                    MATCH (c:Control {id: $ctrl_id})
                    MATCH (p:Process {id: $proc_id})
                    MERGE (c)-[rel:IMPLEMENTS]->(p)
                    SET rel.implementation_level = $implementation_level
                """, ctrl_id=ctrl_id, proc_id=proc_id, implementation_level=properties['implementation_level'])

            logger.info("初始关系创建完成")

    def initialize_knowledge_graph(self):
        """初始化整个知识图谱"""
        try:
            logger.info("开始初始化Neo4j知识图谱...")
            self.create_constraints()
            self.create_indexes()
            self.create_initial_nodes()
            self.create_initial_relationships()
            logger.info("Neo4j知识图谱初始化完成！")
        except Exception as e:
            logger.error(f"知识图谱初始化失败: {e}")
            raise

if __name__ == "__main__":
    neo4j_manager = Neo4jManager()
    try:
        neo4j_manager.initialize_knowledge_graph()
        print("Neo4j知识图谱初始化完成！")
    except Exception as e:
        print(f"Neo4j知识图谱初始化失败: {e}")
    finally:
        neo4j_manager.close()


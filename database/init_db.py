"""
数据库初始化脚本
Database Initialization Script
"""

import pymysql
from config import MYSQL_CONFIG
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_database():
    """创建数据库"""
    try:
        # 连接MySQL服务器（不指定数据库）
        connection = pymysql.connect(
            host=MYSQL_CONFIG['host'],
            port=MYSQL_CONFIG['port'],
            user=MYSQL_CONFIG['user'],
            password=MYSQL_CONFIG['password'],
            charset=MYSQL_CONFIG['charset']
        )

        with connection.cursor() as cursor:
            # 创建数据库
            cursor.execute(
                f"CREATE DATABASE IF NOT EXISTS {MYSQL_CONFIG['database']} CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
            logger.info(f"数据库 {MYSQL_CONFIG['database']} 创建成功")

        connection.close()

    except Exception as e:
        logger.error(f"创建数据库失败: {e}")
        raise


def create_tables():
    """创建所有必要的表"""
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)

        with connection.cursor() as cursor:
            # 1. 审计项目表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_items (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    item_name VARCHAR(255) NOT NULL COMMENT '审计项目名称',
                    item_type ENUM('IT系统', '业务流程', '财务数据', '合规性', '风险控制') NOT NULL COMMENT '审计项目类型',
                    description TEXT COMMENT '项目描述',
                    risk_level ENUM('低', '中', '高', '极高') DEFAULT '中' COMMENT '风险等级',
                    status ENUM('待审计', '审计中', '已完成', '需整改') DEFAULT '待审计' COMMENT '审计状态',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                    INDEX idx_item_type (item_type),
                    INDEX idx_risk_level (risk_level),
                    INDEX idx_status (status)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审计项目表'
            """)

            # 2. 审计标准表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_standards (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    standard_name VARCHAR(255) NOT NULL COMMENT '标准名称',
                    standard_type ENUM('COBIT', 'ISO27001', 'SOX', 'GDPR', '数据安全法', '网络安全法') NOT NULL COMMENT '标准类型',
                    version VARCHAR(50) COMMENT '版本号',
                    description TEXT COMMENT '标准描述',
                    requirements JSON COMMENT '具体要求（JSON格式）',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                    INDEX idx_standard_type (standard_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审计标准表'
            """)

            # 3. 审计结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_results (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    audit_item_id INT NOT NULL COMMENT '审计项目ID',
                    standard_id INT NOT NULL COMMENT '标准ID',
                    compliance_score DECIMAL(5,2) COMMENT '合规性评分(0-100)',
                    risk_score DECIMAL(5,2) COMMENT '风险评分(0-100)',
                    findings TEXT COMMENT '发现的问题',
                    recommendations TEXT COMMENT '整改建议',
                    evidence_files JSON COMMENT '证据文件列表',
                    auditor_id INT COMMENT '审计员ID',
                    audit_date DATE COMMENT '审计日期',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                    FOREIGN KEY (audit_item_id) REFERENCES audit_items(id) ON DELETE CASCADE,
                    FOREIGN KEY (standard_id) REFERENCES audit_standards(id) ON DELETE CASCADE,
                    INDEX idx_audit_item (audit_item_id),
                    INDEX idx_compliance_score (compliance_score),
                    INDEX idx_risk_score (risk_score)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审计结果表'
            """)

            # 4. 知识图谱实体表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_entities (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    entity_id VARCHAR(100) UNIQUE NOT NULL COMMENT '实体ID',
                    entity_name VARCHAR(255) NOT NULL COMMENT '实体名称',
                    entity_type ENUM('组织', '系统', '流程', '风险', '控制', '标准', '法规') NOT NULL COMMENT '实体类型',
                    properties JSON COMMENT '实体属性（JSON格式）',
                    description TEXT COMMENT '实体描述',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                    INDEX idx_entity_type (entity_type),
                    INDEX idx_entity_name (entity_name)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识图谱实体表'
            """)

            # 5. 知识图谱关系表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS knowledge_relations (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    source_entity_id VARCHAR(100) NOT NULL COMMENT '源实体ID',
                    target_entity_id VARCHAR(100) NOT NULL COMMENT '目标实体ID',
                    relation_type VARCHAR(100) NOT NULL COMMENT '关系类型',
                    properties JSON COMMENT '关系属性（JSON格式）',
                    confidence DECIMAL(5,4) DEFAULT 1.0000 COMMENT '置信度',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                    FOREIGN KEY (source_entity_id) REFERENCES knowledge_entities(entity_id) ON DELETE CASCADE,
                    FOREIGN KEY (target_entity_id) REFERENCES knowledge_entities(entity_id) ON DELETE CASCADE,
                    INDEX idx_source_entity (source_entity_id),
                    INDEX idx_target_entity (target_entity_id),
                    INDEX idx_relation_type (relation_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='知识图谱关系表'
            """)

            # 6. 训练数据表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS training_data (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    data_type ENUM('SFT', 'RLHF', 'EVALUATION') NOT NULL COMMENT '数据类型',
                    input_text TEXT NOT NULL COMMENT '输入文本',
                    output_text TEXT COMMENT '输出文本',
                    label_score DECIMAL(5,2) COMMENT '标签评分',
                    metadata JSON COMMENT '元数据',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    INDEX idx_data_type (data_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='训练数据表'
            """)

            # 7. 模型评估结果表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS model_evaluations (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    model_name VARCHAR(255) NOT NULL COMMENT '模型名称',
                    evaluation_type ENUM('BENCHMARK', 'CUSTOM', 'ABLATION') NOT NULL COMMENT '评估类型',
                    metrics JSON NOT NULL COMMENT '评估指标（JSON格式）',
                    test_data_size INT COMMENT '测试数据大小',
                    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '评估时间',
                    notes TEXT COMMENT '备注',
                    INDEX idx_model_name (model_name),
                    INDEX idx_evaluation_type (evaluation_type)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='模型评估结果表'
            """)

            # 8. 审计对话记录表
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS audit_conversations (
                    id INT PRIMARY KEY AUTO_INCREMENT,
                    session_id VARCHAR(100) NOT NULL COMMENT '会话ID',
                    user_input TEXT NOT NULL COMMENT '用户输入',
                    agent_response TEXT NOT NULL COMMENT 'Agent响应',
                    context JSON COMMENT '上下文信息',
                    confidence DECIMAL(5,4) COMMENT '置信度',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                    INDEX idx_session_id (session_id)
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='审计对话记录表'
            """)

            connection.commit()
            logger.info("所有表创建成功")

        connection.close()

    except Exception as e:
        logger.error(f"创建表失败: {e}")
        raise


def insert_initial_data():
    """插入初始数据"""
    try:
        connection = pymysql.connect(**MYSQL_CONFIG)

        with connection.cursor() as cursor:
            # 插入审计标准数据
            standards_data = [
                ('COBIT 2019', 'COBIT', '2019', 'COBIT 2019框架为IT治理和管理提供全面的指导',
                 '{"domains": ["治理", "管理"], "processes": 40, "principles": 5}'),
                ('ISO/IEC 27001:2022', 'ISO27001', '2022', '信息安全管理体系国际标准',
                 '{"controls": 93, "categories": 4, "annexes": 14}'),
                ('SOX法案', 'SOX', '2002', '萨班斯-奥克斯利法案，规范上市公司财务报告',
                 '{"sections": 11, "requirements": ["内部控制", "财务报告", "审计委员会"]}'),
                ('数据安全法', '数据安全法', '2021', '中华人民共和国数据安全法',
                 '{"chapters": 7, "articles": 55, "focus": ["数据分类", "数据保护", "数据跨境"]}'),
                ('网络安全法', '网络安全法', '2017', '中华人民共和国网络安全法',
                 '{"chapters": 7, "articles": 79, "focus": ["网络运行安全", "网络信息安全", "监测预警"]}')
            ]

            cursor.executemany("""
                INSERT INTO audit_standards (standard_name, standard_type, version, description, requirements)
                VALUES (%s, %s, %s, %s, %s)
            """, standards_data)

            # 插入示例审计项目
            audit_items_data = [
                ('ERP系统安全审计', 'IT系统', '对企业资源规划系统的安全性进行全面审计', '高', '待审计'),
                ('财务流程合规性审计', '业务流程', '检查财务流程是否符合SOX法案要求', '中', '待审计'),
                ('数据保护合规审计', '合规性', '评估数据保护措施是否符合数据安全法', '高', '待审计'),
                ('IT基础设施风险评估', '风险控制', '评估IT基础设施的安全风险', '中', '待审计'),
                ('用户权限管理审计', 'IT系统', '审计用户权限分配和管理流程', '中', '待审计')
            ]

            cursor.executemany("""
                INSERT INTO audit_items (item_name, item_type, description, risk_level, status)
                VALUES (%s, %s, %s, %s, %s)
            """, audit_items_data)

            connection.commit()
            logger.info("初始数据插入成功")

        connection.close()

    except Exception as e:
        logger.error(f"插入初始数据失败: {e}")
        raise


if __name__ == "__main__":
    try:
        create_database()
        create_tables()
        insert_initial_data()
        print("数据库初始化完成！")
    except Exception as e:
        print(f"数据库初始化失败: {e}")


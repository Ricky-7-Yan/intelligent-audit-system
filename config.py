"""
智能审计决策系统 - 主配置文件
Intelligent Audit Decision System - Main Configuration
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('config.env')

# 项目根目录
PROJECT_ROOT = Path(__file__).parent

# 数据库配置
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST', 'localhost'),
    'port': int(os.getenv('MYSQL_PORT', 3306)),
    'user': os.getenv('MYSQL_USER', 'root'),
    'password': os.getenv('MYSQL_PASSWORD', '123456'),
    'database': os.getenv('MYSQL_DATABASE', 'audit_system'),
    'charset': 'utf8mb4'
}

# Neo4j配置
NEO4J_CONFIG = {
    'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
    'user': os.getenv('NEO4J_USER', 'neo4j'),
    'password': os.getenv('NEO4J_PASSWORD', '12345678')
}

# LLM配置
LLM_CONFIG = {
    'api_key': os.getenv('QWEN_API_KEY', 'sk-484fb339d2274307b3aa3fd6400964ae'),
    'base_url': os.getenv('QWEN_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'),
    'model': 'qwen-turbo',
    'max_tokens': int(os.getenv('MAX_TOKENS', 4096)),
    'temperature': float(os.getenv('TEMPERATURE', 0.7)),
    'top_p': float(os.getenv('TOP_P', 0.9))
}

# 路径配置
PATHS = {
    'data': PROJECT_ROOT / 'data',
    'training_data': PROJECT_ROOT / 'data' / 'training',
    'models': PROJECT_ROOT / 'models',
    'logs': PROJECT_ROOT / 'logs',
    'static': PROJECT_ROOT / 'static',
    'templates': PROJECT_ROOT / 'templates'
}

# 创建必要的目录
for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Web服务配置
WEB_CONFIG = {
    'host': os.getenv('WEB_HOST', '0.0.0.0'),
    'port': int(os.getenv('WEB_PORT', 8000)),
    'debug': os.getenv('DEBUG', 'True').lower() == 'true'
}

# 审计相关配置
AUDIT_CONFIG = {
    'max_audit_items': 1000,
    'risk_threshold': 0.7,
    'compliance_check_interval': 3600,  # 1小时
    'knowledge_graph_update_interval': 86400  # 24小时
}

# 训练配置
TRAINING_CONFIG = {
    'batch_size': 8,
    'learning_rate': 2e-5,
    'num_epochs': 3,
    'warmup_steps': 100,
    'max_grad_norm': 1.0,
    'save_steps': 500,
    'eval_steps': 100
}


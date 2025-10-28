#!/usr/bin/env python3
"""
智能审计决策系统启动脚本
Intelligent Audit Decision System Startup Script
"""

import os
import sys
import subprocess
import time
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查依赖包"""
    logger.info("检查依赖包...")

    required_packages = [
        'torch', 'transformers', 'langchain', 'langgraph',
        'neo4j', 'pymysql', 'fastapi', 'uvicorn',
        'spacy', 'networkx', 'chromadb',
        'sentence_transformers', 'faiss'
    ]

    missing_packages = []

    for package in required_packages:
        try:
            module_name = package.replace('-', '_')
            __import__(module_name)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logger.error(f"缺少依赖包: {', '.join(missing_packages)}")
        logger.info("请运行: pip install -r requirements.txt")
        return False

    logger.info("依赖包检查完成")
    return True


def check_directories():
    """检查并创建必要的目录"""
    logger.info("检查目录结构...")

    from config import PATHS

    required_dirs = [
        PATHS['data'],
        PATHS['training_data'],
        PATHS['models'],
        PATHS['logs'],
        PATHS['static'],
        PATHS['templates']
    ]

    for dir_path in required_dirs:
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"目录已创建: {dir_path}")

    return True


def start_web_server():
    """启动Web服务器"""
    logger.info("启动Web服务器...")

    try:
        import uvicorn
        from web.main import app

        from config import WEB_CONFIG
        uvicorn.run(
            app,
            host=WEB_CONFIG['host'],
            port=WEB_CONFIG['port']
        )

    except Exception as e:
        logger.error(f"Web服务器启动失败: {e}")
        return False

    return True


def main():
    """主函数"""
    logger.info("智能审计决策系统启动中...")

    # 检查依赖
    if not check_dependencies():
        logger.warning("部分依赖缺失，但将继续启动...")

    # 检查目录
    if not check_directories():
        sys.exit(1)

    # 启动Web服务器
    logger.info("=" * 50)
    logger.info("系统启动完成！")
    logger.info("访问地址: http://localhost:8000")
    logger.info("=" * 50)
    start_web_server()


if __name__ == "__main__":
    main()

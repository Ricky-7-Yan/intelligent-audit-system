#!/usr/bin/env python3
"""Start the Intelligent Audit System web server."""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def check_dependencies() -> bool:
    required_modules = [
        "fastapi",
        "uvicorn",
        "jinja2",
        "pydantic",
        "langchain_core",
        "langchain_openai",
        "numpy",
        "sklearn",
        "networkx",
        "dotenv",
    ]
    optional_modules = ["sentence_transformers", "neo4j", "pymysql"]

    missing = [module for module in required_modules if not _can_import(module)]
    optional_missing = [module for module in optional_modules if not _can_import(module)]

    if missing:
        logger.error("缺少必需依赖: %s", ", ".join(missing))
        logger.error("请运行: pip install -r requirements.txt")
        return False

    if optional_missing:
        logger.warning("缺少可选增强依赖: %s。系统将使用降级能力继续启动。", ", ".join(optional_missing))

    return True


def _can_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


def check_directories() -> None:
    from config import PATHS

    for path in PATHS.values():
        path.mkdir(parents=True, exist_ok=True)
        logger.info("目录已就绪: %s", path)


def start_web_server() -> None:
    import uvicorn

    from config import WEB_CONFIG
    from web.main import app

    logger.info("系统启动完成，访问地址: http://localhost:%s", WEB_CONFIG["port"])
    uvicorn.run(app, host=WEB_CONFIG["host"], port=WEB_CONFIG["port"])


def main() -> None:
    logger.info("智能审计 Agent 平台启动中...")
    if not check_dependencies():
        sys.exit(1)
    check_directories()
    start_web_server()


if __name__ == "__main__":
    main()

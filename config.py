"""
Application configuration for the Intelligent Audit System.

The module intentionally keeps secrets outside source control. Runtime values are
loaded from ``config.env`` when present, then from the process environment.
"""

import ipaddress
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / "config.env")


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _list_env(name: str, default: str = "") -> List[str]:
    raw = os.getenv(name, default)
    return [item.strip() for item in raw.split(",") if item.strip()]


def _json_env(name: str, default: Any) -> Any:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default


PATHS: Dict[str, Path] = {
    "data": PROJECT_ROOT / "data",
    "training_data": PROJECT_ROOT / "data" / "training",
    "models": PROJECT_ROOT / "models",
    "logs": PROJECT_ROOT / "logs",
    "static": PROJECT_ROOT / "static",
    "templates": PROJECT_ROOT / "templates",
    "uploads": PROJECT_ROOT / "data" / "uploads",
    "rag_store": PROJECT_ROOT / "data" / "rag_store",
    "evaluation_runs": PROJECT_ROOT / "data" / "evaluation_runs",
    "evidence_analyses": PROJECT_ROOT / "data" / "evidence_analyses",
    "agent_runtime": PROJECT_ROOT / "data" / "agent_runtime",
    "audit_events": PROJECT_ROOT / "data" / "audit_events",
    "quarantine": PROJECT_ROOT / "data" / "quarantine",
}

for path in PATHS.values():
    path.mkdir(parents=True, exist_ok=True)


MYSQL_CONFIG: Dict[str, Any] = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": _int_env("MYSQL_PORT", 3306),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", ""),
    "database": os.getenv("MYSQL_DATABASE", "audit_system"),
    "charset": "utf8mb4",
    "connect_timeout": _int_env("MYSQL_CONNECT_TIMEOUT", 3),
}

NEO4J_CONFIG: Dict[str, Any] = {
    "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
    "user": os.getenv("NEO4J_USER", "neo4j"),
    "password": os.getenv("NEO4J_PASSWORD", ""),
    "timeout": _int_env("NEO4J_CONNECT_TIMEOUT", 3),
}

LLM_CONFIG: Dict[str, Any] = {
    "provider": os.getenv("LLM_PROVIDER", "deepseek"),
    "api_key": os.getenv("DEEPSEEK_API_KEY") or os.getenv("QWEN_API_KEY") or os.getenv("OPENAI_API_KEY") or "",
    "base_url": os.getenv(
        "LLM_BASE_URL",
        os.getenv("DEEPSEEK_BASE_URL", os.getenv("QWEN_BASE_URL", "https://api.deepseek.com")),
    ),
    "model": os.getenv("LLM_MODEL", os.getenv("DEEPSEEK_MODEL", os.getenv("QWEN_MODEL", "deepseek-chat"))),
    "max_tokens": _int_env("MAX_TOKENS", 2048),
    "temperature": _float_env("TEMPERATURE", 0.2),
    "top_p": _float_env("TOP_P", 0.9),
}
LLM_CONFIG["enabled"] = bool(LLM_CONFIG["api_key"])

WEB_CONFIG: Dict[str, Any] = {
    "host": os.getenv("WEB_HOST", "127.0.0.1"),
    "port": _int_env("PORT", _int_env("WEB_PORT", 8000)),
    "debug": _bool_env("DEBUG", False),
    "cors_origins": _list_env("CORS_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"),
}

SECURITY_CONFIG: Dict[str, Any] = {
    # local keeps the zero-config desktop experience. Set enforced in any shared deployment.
    "mode": os.getenv("SECURITY_MODE", "local").strip().lower(),
    "default_tenant": os.getenv("DEFAULT_TENANT_ID", "local"),
    "default_subject": os.getenv("DEFAULT_SUBJECT", "local-admin"),
    "api_tokens": _json_env("AUDITPILOT_API_TOKENS_JSON", {}),
    "rate_limit_per_minute": _int_env("API_RATE_LIMIT_PER_MINUTE", 180),
    "tenant_isolation": _bool_env("TENANT_ISOLATION", True),
    "audit_log_signing_key": os.getenv("AUDIT_LOG_SIGNING_KEY", ""),
}

UPLOAD_CONFIG: Dict[str, Any] = {
    "knowledge_max_bytes": _int_env("KNOWLEDGE_UPLOAD_MAX_BYTES", 5 * 1024 * 1024),
    "evidence_max_bytes": _int_env("EVIDENCE_UPLOAD_MAX_BYTES", 5 * 1024 * 1024),
    "chunk_bytes": _int_env("UPLOAD_CHUNK_BYTES", 64 * 1024),
    "reject_prompt_injection": _bool_env("REJECT_PROMPT_INJECTION", True),
}

AUDIT_CONFIG: Dict[str, Any] = {
    "max_session_messages": _int_env("MAX_SESSION_MESSAGES", 20),
    "risk_threshold_high": _float_env("RISK_THRESHOLD_HIGH", 0.72),
    "risk_threshold_medium": _float_env("RISK_THRESHOLD_MEDIUM", 0.42),
}

RAG_CONFIG: Dict[str, Any] = {
    "chunk_size": _int_env("RAG_CHUNK_SIZE", 700),
    "chunk_overlap": _int_env("RAG_CHUNK_OVERLAP", 120),
    "top_k": _int_env("RAG_TOP_K", 5),
    "store_file": PATHS["rag_store"] / "documents.json",
    "embedding_model": os.getenv(
        "RAG_EMBEDDING_MODEL",
        str(PATHS["models"] / "sentence-transformers" / "paraphrase-multilingual-MiniLM-L12-v2"),
    ),
    "enable_tfidf": _bool_env("RAG_ENABLE_TFIDF", True),
    "enable_embeddings": _bool_env("RAG_ENABLE_EMBEDDINGS", False),
}

TRAINING_CONFIG: Dict[str, Any] = {
    "batch_size": _int_env("TRAINING_BATCH_SIZE", 8),
    "learning_rate": _float_env("TRAINING_LEARNING_RATE", 2e-5),
    "num_epochs": _int_env("TRAINING_EPOCHS", 3),
    "max_grad_norm": _float_env("TRAINING_MAX_GRAD_NORM", 1.0),
}


def runtime_configuration_issues() -> List[str]:
    """Return unsafe runtime combinations that block readiness and startup."""

    issues: List[str] = []
    mode = str(SECURITY_CONFIG.get("mode") or "local").lower()
    host = str(WEB_CONFIG.get("host") or "")
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = host.lower() == "localhost"
    if mode == "local" and not is_loopback:
        issues.append("SECURITY_MODE=local 只能绑定 127.0.0.1、::1 或 localhost")
    if mode == "enforced":
        if not SECURITY_CONFIG.get("api_tokens"):
            issues.append("SECURITY_MODE=enforced 必须配置至少一个 API token")
        if WEB_CONFIG.get("debug"):
            issues.append("SECURITY_MODE=enforced 禁止启用 DEBUG")
        if "*" in set(WEB_CONFIG.get("cors_origins") or []):
            issues.append("SECURITY_MODE=enforced 禁止使用通配 CORS")
        if not SECURITY_CONFIG.get("audit_log_signing_key"):
            issues.append("SECURITY_MODE=enforced 必须配置 AUDIT_LOG_SIGNING_KEY")
    return issues


def validate_runtime_configuration() -> None:
    issues = runtime_configuration_issues()
    if issues:
        raise RuntimeError("；".join(issues))

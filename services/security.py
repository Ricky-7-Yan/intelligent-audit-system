"""Request authentication, authorization, tenant context and tamper-evident audit events."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, Optional

from config import PATHS, SECURITY_CONFIG


ROLE_PERMISSIONS: Dict[str, set[str]] = {
    "system": {"system:*"},
    "admin": {"*"},
    "audit_manager": {
        "read:*",
        "write:audit",
        "write:agent",
        "write:evidence",
        "write:knowledge",
        "write:evaluation",
        "execute:skill",
        "review:*",
        "delete:*",
        "export:audit",
    },
    "auditor": {
        "read:*",
        "write:audit",
        "write:agent",
        "write:evidence",
        "write:knowledge",
        "write:evaluation",
        "execute:skill",
        "export:audit",
    },
    "reviewer": {"read:*", "review:*", "export:audit"},
    "remediation_owner": {"read:audit", "read:agent", "write:audit"},
    "reader": {"read:*"},
}


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: tuple[str, ...]
    permissions: frozenset[str]
    auth_method: str
    project_ids: tuple[str, ...] = ()

    def public_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["permissions"] = sorted(self.permissions)
        payload["roles"] = list(self.roles)
        payload["project_ids"] = list(self.project_ids)
        return payload


SYSTEM_PRINCIPAL = Principal(
    subject="system",
    tenant_id=str(SECURITY_CONFIG["default_tenant"]),
    roles=("system", "admin"),
    permissions=frozenset({"system:*", "*"}),
    auth_method="internal",
    project_ids=("*",),
)
_principal_context: ContextVar[Principal] = ContextVar("auditpilot_principal", default=SYSTEM_PRINCIPAL)
_request_id_context: ContextVar[str] = ContextVar("auditpilot_request_id", default="")


def current_principal() -> Principal:
    return _principal_context.get()


def current_tenant_id() -> str:
    return current_principal().tenant_id


def current_request_id() -> str:
    return _request_id_context.get()


def record_visible(record: Dict[str, Any]) -> bool:
    legacy_tenant = str(SECURITY_CONFIG["default_tenant"])
    return str(record.get("tenant_id") or legacy_tenant) == current_tenant_id()


def project_visible(record: Dict[str, Any], project_id: str = "") -> bool:
    """Apply tenant and project membership checks to a persisted record."""

    if not record_visible(record):
        return False
    principal = current_principal()
    if principal.auth_method in {"local", "internal"}:
        return True
    target = str(project_id or record.get("project_id") or record.get("run_id") or "")
    if not target:
        return True
    if "*" in principal.project_ids or target in principal.project_ids:
        return True
    members = record.get("members") or {}
    if isinstance(members, dict) and principal.subject in members:
        return True
    if isinstance(members, list):
        return principal.subject in {
            str(item.get("subject") if isinstance(item, dict) else item) for item in members
        }
    return False


def bind_request_context(principal: Principal, request_id: str) -> tuple[Token[Principal], Token[str]]:
    return _principal_context.set(principal), _request_id_context.set(request_id)


def reset_request_context(tokens: tuple[Token[Principal], Token[str]]) -> None:
    _principal_context.reset(tokens[0])
    _request_id_context.reset(tokens[1])


def _permissions_for_roles(roles: Iterable[str]) -> set[str]:
    permissions: set[str] = set()
    for role in roles:
        permissions.update(ROLE_PERMISSIONS.get(role, set()))
    return permissions


def authenticate(authorization: str = "", *, allow_anonymous: bool = False) -> Principal:
    """Resolve a bearer token from secret-managed configuration.

    In local mode the desktop application remains zero-config. Enforced mode
    rejects missing or unknown credentials.
    """

    mode = str(SECURITY_CONFIG.get("mode") or "local").lower()
    token: Optional[str] = None
    if authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    configured = SECURITY_CONFIG.get("api_tokens") or {}
    record = configured.get(token) if token else None
    if isinstance(record, dict):
        roles = tuple(str(item) for item in record.get("roles", ["reader"]))
        explicit = {str(item) for item in record.get("permissions", [])}
        return Principal(
            subject=str(record.get("subject") or "api-user"),
            tenant_id=_safe_tenant(str(record.get("tenant_id") or SECURITY_CONFIG["default_tenant"])),
            roles=roles,
            permissions=frozenset(_permissions_for_roles(roles) | explicit),
            auth_method="bearer",
            project_ids=tuple(str(item) for item in record.get("project_ids", [])),
        )
    if mode == "local":
        return Principal(
            subject=str(SECURITY_CONFIG["default_subject"]),
            tenant_id=_safe_tenant(str(SECURITY_CONFIG["default_tenant"])),
            roles=("admin",),
            permissions=frozenset({"*"}),
            auth_method="local",
            project_ids=("*",),
        )
    if allow_anonymous:
        return Principal(
            subject="anonymous",
            tenant_id=_safe_tenant(str(SECURITY_CONFIG["default_tenant"])),
            roles=(),
            permissions=frozenset(),
            auth_method="anonymous",
        )
    raise PermissionError("缺少或无效的 Bearer 凭据")


def has_permission(principal: Principal, required: str) -> bool:
    if required.startswith("system:") and "system" not in principal.roles:
        return False
    if "*" in principal.permissions or required in principal.permissions:
        return True
    domain = required.split(":", 1)[-1]
    action = required.split(":", 1)[0]
    return f"{action}:*" in principal.permissions or f"*:{domain}" in principal.permissions


def required_permission(method: str, path: str) -> Optional[str]:
    if not path.startswith("/api/") or path.startswith(("/api/health", "/api/security/session")):
        return None
    domain = _api_domain(path)
    if method.upper() == "GET":
        return f"read:{domain}"
    if method.upper() == "DELETE":
        return f"delete:{domain}"
    if "/review" in path:
        return f"review:{domain}"
    if "/delivery" in path or "/report" in path:
        return f"export:{domain}"
    if path.startswith("/api/skills/") and path.endswith("/run"):
        return "execute:skill"
    return f"write:{domain}"


def _api_domain(path: str) -> str:
    segment = path.split("/", 3)[2] if path.count("/") >= 2 else "system"
    return {
        "training": "evaluation",
        "session": "memory",
        "mcp": "skill",
        "product": "audit",
        "research": "knowledge",
        "search": "audit",
        "safety": "agent",
    }.get(segment, segment)


def _safe_tenant(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "-", value.strip())[:80]
    return cleaned or "default"


class SlidingWindowRateLimiter:
    def __init__(self, limit: int) -> None:
        self.limit = max(10, int(limit))
        self._requests: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - 60
        with self._lock:
            queue = self._requests[key]
            while queue and queue[0] < cutoff:
                queue.popleft()
            if len(queue) >= self.limit:
                retry_after = max(1, int(60 - (now - queue[0])))
                return False, retry_after
            queue.append(now)
            return True, 0


class AuditEventStore:
    """Append-only hash chain with an optional HMAC-signed head checkpoint."""

    def __init__(self, path: Optional[Path] = None, signing_key: Optional[str] = None) -> None:
        self.path = path or (PATHS["audit_events"] / "events.jsonl")
        self.checkpoint_path = self.path.with_suffix(".checkpoint.json")
        configured_key = SECURITY_CONFIG.get("audit_log_signing_key", "")
        self._signing_key = str(configured_key if signing_key is None else signing_key).encode("utf-8")
        self._event_count: Optional[int] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def append(self, action: str, resource: str, status: int, details: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        with self._lock:
            previous_hash = self._last_hash()
            principal = current_principal()
            event = {
                "event_id": f"AE-{uuid.uuid4().hex[:12].upper()}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "request_id": current_request_id(),
                "tenant_id": principal.tenant_id,
                "subject": principal.subject,
                "roles": list(principal.roles),
                "action": action,
                "resource": resource,
                "status": int(status),
                "details": details or {},
                "previous_hash": previous_hash,
            }
            event["event_hash"] = self._digest(event)
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
            if self._signing_key:
                if self._event_count is None:
                    self._event_count = sum(
                        1 for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()
                    )
                else:
                    self._event_count += 1
                self._write_checkpoint(event["event_hash"], self._event_count)
            return event

    def verify(self) -> Dict[str, Any]:
        previous = ""
        checked = 0
        if not self.path.exists():
            return {
                "valid": True,
                "events": 0,
                "head_hash": "",
                "checkpoint": "disabled" if not self._signing_key else "missing",
            }
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            expected = event.pop("event_hash", "")
            if event.get("previous_hash", "") != previous or self._digest(event) != expected:
                return {"valid": False, "events": checked, "head_hash": previous}
            previous = expected
            checked += 1
        checkpoint_status = self._verify_checkpoint(previous, checked)
        self._event_count = checked
        return {
            "valid": checkpoint_status in {"valid", "disabled"},
            "events": checked,
            "head_hash": previous,
            "checkpoint": checkpoint_status,
        }

    def _last_hash(self) -> str:
        if not self.path.exists():
            return ""
        lines = [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            return ""
        try:
            return str(json.loads(lines[-1]).get("event_hash") or "")
        except json.JSONDecodeError:
            return "CORRUPT"

    @staticmethod
    def _digest(event: Dict[str, Any]) -> str:
        raw = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _write_checkpoint(self, head_hash: str, event_count: int) -> None:
        checkpoint = {
            "head_hash": head_hash,
            "event_count": event_count,
            "signed_at": datetime.now(timezone.utc).isoformat(),
            "algorithm": "HMAC-SHA256",
        }
        checkpoint["signature"] = self._checkpoint_signature(checkpoint)
        temporary = self.checkpoint_path.with_suffix(".checkpoint.tmp")
        temporary.write_text(json.dumps(checkpoint, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        temporary.replace(self.checkpoint_path)

    def _verify_checkpoint(self, head_hash: str, event_count: int) -> str:
        if not self._signing_key:
            return "disabled"
        if not self.checkpoint_path.exists():
            return "missing"
        try:
            checkpoint = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return "invalid"
        expected = str(checkpoint.pop("signature", ""))
        actual = self._checkpoint_signature(checkpoint)
        if not hmac.compare_digest(expected, actual):
            return "invalid"
        if checkpoint.get("head_hash") != head_hash or int(checkpoint.get("event_count") or -1) != event_count:
            return "stale"
        return "valid"

    def _checkpoint_signature(self, checkpoint: Dict[str, Any]) -> str:
        raw = json.dumps(checkpoint, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hmac.new(self._signing_key, raw.encode("utf-8"), hashlib.sha256).hexdigest()

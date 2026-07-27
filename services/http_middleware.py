"""HTTP middleware for request identity, API auth, RBAC, rate limits and audit logs."""

from __future__ import annotations

import uuid
from typing import Callable

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from config import SECURITY_CONFIG
from services.security import (
    AuditEventStore,
    SlidingWindowRateLimiter,
    authenticate,
    bind_request_context,
    has_permission,
    required_permission,
    reset_request_context,
)


class SecurityAuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, audit_events: AuditEventStore | None = None) -> None:
        super().__init__(app)
        self.audit_events = audit_events or AuditEventStore()
        self.rate_limiter = SlidingWindowRateLimiter(int(SECURITY_CONFIG["rate_limit_per_minute"]))

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = _request_id(request.headers.get("x-request-id", ""))
        public_request = (
            not request.url.path.startswith("/api/")
            or request.url.path.startswith("/api/health")
        )
        try:
            principal = authenticate(
                request.headers.get("authorization", ""),
                allow_anonymous=public_request,
            )
        except PermissionError as exc:
            return _json_error(401, str(exc), request_id, {"WWW-Authenticate": "Bearer"})

        tokens = bind_request_context(principal, request_id)
        request.state.principal = principal
        request.state.request_id = request_id
        permission = required_permission(request.method, request.url.path)
        try:
            if permission and not has_permission(principal, permission):
                self.audit_events.append(
                    "authorization_denied",
                    request.url.path,
                    403,
                    {"method": request.method, "required_permission": permission},
                )
                return _json_error(403, f"缺少权限：{permission}", request_id)

            if request.url.path.startswith("/api/"):
                allowed, retry_after = self.rate_limiter.allow(f"{principal.tenant_id}:{principal.subject}")
                if not allowed:
                    self.audit_events.append(
                        "rate_limit_exceeded",
                        request.url.path,
                        429,
                        {"method": request.method, "retry_after": retry_after},
                    )
                    return _json_error(429, "请求过于频繁，请稍后重试", request_id, {"Retry-After": str(retry_after)})

            try:
                response = await call_next(request)
            except Exception:
                self.audit_events.append(
                    "request_failed",
                    request.url.path,
                    500,
                    {"method": request.method},
                )
                raise

            if request.url.path.startswith("/api/") and (
                request.method.upper() in {"POST", "PUT", "PATCH", "DELETE"} or response.status_code >= 400
            ):
                self.audit_events.append(
                    f"http_{request.method.lower()}",
                    request.url.path,
                    response.status_code,
                    {"permission": permission or "public"},
                )
            _security_headers(response, request_id)
            return response
        finally:
            reset_request_context(tokens)


def _request_id(value: str) -> str:
    cleaned = "".join(character for character in value if character.isalnum() or character in "-_.")[:96]
    return cleaned or f"REQ-{uuid.uuid4().hex[:16].upper()}"


def _json_error(status: int, detail: str, request_id: str, headers: dict[str, str] | None = None) -> JSONResponse:
    response = JSONResponse({"success": False, "detail": detail, "request_id": request_id}, status_code=status)
    for key, value in (headers or {}).items():
        response.headers[key] = value
    _security_headers(response, request_id)
    return response


def _security_headers(response: Response, request_id: str) -> None:
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"

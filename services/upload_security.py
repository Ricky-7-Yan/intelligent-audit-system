"""Streaming upload validation and document prompt-injection screening."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from fastapi import UploadFile

from config import UPLOAD_CONFIG


PROMPT_INJECTION_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"disregard\s+(the\s+)?system\s+prompt",
        r"reveal\s+(the\s+)?system\s+prompt",
        r"developer\s+message",
        r"忽略(以上|此前|之前).{0,12}(指令|提示词)",
        r"(泄露|输出|显示).{0,12}(系统提示词|开发者指令)",
        r"绕过.{0,8}(权限|安全|审核)",
    )
]
SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"Bearer\s+[A-Za-z0-9._-]{16,}", re.IGNORECASE),
]
PERSONAL_DATA_PATTERNS = [
    re.compile(r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"),
    re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"),
    re.compile(r"(?<!\d)\d{16,19}(?!\d)"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
]
BUSINESS_CONFIDENTIAL_PATTERNS = [
    re.compile(r"(内部资料|仅限内部|商业机密|商业秘密|保密协议|未公开|涉密)"),
    re.compile(
        r"(?:confidential|internal\s+only|internal[-_]use|proprietary|trade\s+secret|do\s+not\s+distribute)",
        re.IGNORECASE,
    ),
    # Four octets only, so a standard clause such as "ISO 27001 10.1.2" is not a hit.
    re.compile(
        r"(?<![\d.])(?:10\.\d{1,3}|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d{1,3}\.\d{1,3}(?![\d.])"
    ),
    re.compile(r"[A-Za-z0-9-]+\.(?:internal|intranet|corp|lan)(?![A-Za-z0-9-])", re.IGNORECASE),
]
# One screening vocabulary shared by file uploads and outbound research queries.
SENSITIVE_PATTERN_GROUPS: Dict[str, list[re.Pattern[str]]] = {
    "credential": SECRET_PATTERNS,
    "personal_data": PERSONAL_DATA_PATTERNS,
    "business_confidential": BUSINESS_CONFIDENTIAL_PATTERNS,
    "prompt_injection": PROMPT_INJECTION_PATTERNS,
}


def scan_sensitive_text(text: str) -> Dict[str, list[str]]:
    """Group the matching pattern sources by category without echoing the input."""

    return {
        category: [pattern.pattern for pattern in patterns if pattern.search(text)]
        for category, patterns in SENSITIVE_PATTERN_GROUPS.items()
    }


@dataclass
class ValidatedUpload:
    content: bytes
    original_name: str
    suffix: str
    sha256: str
    size_bytes: int
    content_type: str
    security: Dict[str, Any]


async def read_validated_upload(
    file: UploadFile,
    *,
    allowed_extensions: Iterable[str],
    max_bytes: int,
) -> ValidatedUpload:
    original_name = Path(file.filename or "upload").name
    suffix = Path(original_name).suffix.lower()
    allowed = {item.lower() for item in allowed_extensions}
    if suffix not in allowed:
        raise ValueError(f"不支持的文件类型：{suffix or '无扩展名'}")

    chunks: list[bytes] = []
    size = 0
    digest = hashlib.sha256()
    chunk_size = max(4096, int(UPLOAD_CONFIG["chunk_bytes"]))
    while True:
        chunk = await file.read(chunk_size)
        if not chunk:
            break
        size += len(chunk)
        if size > int(max_bytes):
            raise ValueError(f"文件超过 {int(max_bytes) // (1024 * 1024)}MB 限制")
        chunks.append(chunk)
        digest.update(chunk)
    content = b"".join(chunks)
    if b"\x00" in content[:8192]:
        raise ValueError("检测到二进制内容，文本上传已拒绝")
    if content.startswith((b"PK\x03\x04", b"%PDF", b"\x89PNG", b"\xff\xd8\xff")):
        raise ValueError("文件内容与文本扩展名不一致")
    text = content.decode("utf-8", errors="replace")
    findings = scan_sensitive_text(text)
    injection_hits = findings["prompt_injection"]
    secret_hits = findings["credential"]
    return ValidatedUpload(
        content=content,
        original_name=original_name,
        suffix=suffix,
        sha256=digest.hexdigest(),
        size_bytes=size,
        content_type=file.content_type or "application/octet-stream",
        security={
            "prompt_injection_detected": bool(injection_hits),
            "prompt_injection_signals": injection_hits,
            "credential_like_content_detected": bool(secret_hits),
            "credential_signal_count": len(secret_hits),
        },
    )

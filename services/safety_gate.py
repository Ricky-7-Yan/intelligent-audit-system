"""Runtime safety checks for audit agent tasks and tool calls."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List


SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?[^'\"\s,;]{8,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
]

DESTRUCTIVE_TERMS = [
    "delete database",
    "drop table",
    "truncate table",
    "remove all",
    "erase evidence",
    "bypass approval",
    "disable audit",
]

EVIDENCE_TERMS = ["evidence", "source", "sample", "log", "approval", "screenshot", "workpaper"]


@dataclass
class GateFinding:
    code: str
    level: str
    message: str
    field: str = ""


@dataclass
class SafetyGateResult:
    status: str
    score: float
    findings: List[GateFinding] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "score": round(self.score, 3),
            "findings": [finding.__dict__ for finding in self.findings],
        }


class SafetyGate:
    """Deterministic policy gate used before an agent executes tools."""

    def inspect(self, payload: Dict[str, Any], stage: str = "runtime") -> Dict[str, Any]:
        text = self._flatten(payload)
        findings: List[GateFinding] = []

        for pattern in SECRET_PATTERNS:
            if pattern.search(text):
                findings.append(
                    GateFinding(
                        code="secret_leak_risk",
                        level="block",
                        message="Input appears to contain an API key, token, or credential. Remove secrets before execution.",
                        field=stage,
                    )
                )
                break

        lowered = text.lower()
        for term in DESTRUCTIVE_TERMS:
            if term in lowered:
                findings.append(
                    GateFinding(
                        code="destructive_action",
                        level="block",
                        message=f"Requested action includes a destructive operation: {term}.",
                        field=stage,
                    )
                )

        if stage in {"audit", "evidence", "runtime"} and len(text) > 20:
            evidence_hits = sum(1 for term in EVIDENCE_TERMS if term in lowered)
            if evidence_hits == 0:
                findings.append(
                    GateFinding(
                        code="evidence_gap",
                        level="warn",
                        message="No explicit evidence/source/sample signal found. Keep this step under human review.",
                        field=stage,
                    )
                )

        block_count = sum(1 for finding in findings if finding.level == "block")
        warn_count = sum(1 for finding in findings if finding.level == "warn")
        score = max(0.0, 1.0 - block_count * 0.55 - warn_count * 0.12)
        status = "blocked" if block_count else "review" if warn_count else "pass"
        return SafetyGateResult(status=status, score=score, findings=findings).to_dict()

    def _flatten(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            return " ".join(f"{key} {self._flatten(item)}" for key, item in value.items())
        if isinstance(value, (list, tuple, set)):
            return " ".join(self._flatten(item) for item in value)
        return str(value)

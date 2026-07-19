"""Durable governance and acceptance gates for bounded Agent evolution."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import PATHS, PROJECT_ROOT


DEFAULT_SURFACES: Dict[str, List[str]] = {
    "locked": [
        "services/evaluation_repository.py",
        "training/training_pipeline.py",
        "tests/test_agent_platform.py",
    ],
    "editable": [
        "agents/",
        "rag/",
        "services/",
        "training/",
        "static/",
        "templates/",
    ],
    "append_only": [
        "data/harness/events.jsonl",
        "data/harness/rejected.jsonl",
        "data/skill_runs/runs.jsonl",
    ],
    "human_controlled": [
        "production promotion",
        "evaluator changes",
        "credential access",
        "destructive operations",
    ],
}


class HarnessControlPlane:
    """Keeps the evaluator outside the self-improvement loop.

    Candidates can be proposed and measured automatically, but a passing
    candidate only reaches ``awaiting_human_review``. Promotion remains an
    explicit human decision and every state transition is append-only.
    """

    def __init__(self, base_dir: Optional[Path] = None, project_root: Optional[Path] = None) -> None:
        self.base_dir = base_dir or PATHS["data"] / "harness"
        self.project_root = project_root or PROJECT_ROOT
        self.candidate_dir = self.base_dir / "candidates"
        self.archive_dir = self.base_dir / "archive"
        self.events_file = self.base_dir / "events.jsonl"
        self.rejected_file = self.base_dir / "rejected.jsonl"
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def surfaces(self) -> Dict[str, Any]:
        return {
            **DEFAULT_SURFACES,
            "locked_fingerprints": self._fingerprints(DEFAULT_SURFACES["locked"]),
            "policy": {
                "max_editable_surfaces_per_candidate": 1,
                "no_regression_on_held_in": True,
                "no_regression_on_held_out": True,
                "strict_improvement_required": True,
                "automatic_promotion": False,
            },
        }

    def create_candidate(
        self,
        proposal: Dict[str, Any],
        editable_surface: str = "services/",
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not self._is_editable(editable_surface):
            raise ValueError(f"editable_surface 不在允许范围内：{editable_surface}")
        now = datetime.now().isoformat()
        candidate_id = f"HC-{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
        record = {
            "candidate_id": candidate_id,
            "proposal_id": proposal.get("proposal_id"),
            "title": proposal.get("title") or "未命名 Harness 候选",
            "hypothesis": proposal.get("action") or proposal.get("hypothesis") or "",
            "validation_plan": proposal.get("validation") or "",
            "editable_surface": editable_surface,
            "parent_id": parent_id,
            "status": "draft",
            "locked_fingerprints": self._fingerprints(DEFAULT_SURFACES["locked"]),
            "scores": {},
            "checks": [],
            "decision": None,
            "created_at": now,
            "updated_at": now,
        }
        with self._lock:
            self._write_candidate(record)
            self._append_event("candidate_created", candidate_id, {"proposal_id": record["proposal_id"], "editable_surface": editable_surface})
        return record

    def evaluate_candidate(
        self,
        candidate_id: str,
        baseline: Dict[str, float],
        held_in: Dict[str, float],
        held_out: Dict[str, float],
        checks: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        locked_unchanged = candidate.get("locked_fingerprints") == self._fingerprints(DEFAULT_SURFACES["locked"])
        dimensions = sorted(set(baseline) | set(held_in) | set(held_out))
        deltas = {
            name: {
                "held_in": round(float(held_in.get(name, 0)) - float(baseline.get(name, 0)), 4),
                "held_out": round(float(held_out.get(name, 0)) - float(baseline.get(name, 0)), 4),
            }
            for name in dimensions
        }
        no_regression = all(item["held_in"] >= 0 and item["held_out"] >= 0 for item in deltas.values())
        strict_improvement = any(item["held_in"] > 0 or item["held_out"] > 0 for item in deltas.values())
        deterministic_checks = checks or []
        checks_passed = all(item.get("status") == "pass" for item in deterministic_checks) if deterministic_checks else True
        accepted = locked_unchanged and no_regression and strict_improvement and checks_passed
        candidate.update(
            {
                "status": "awaiting_human_review" if accepted else "rejected",
                "scores": {"baseline": baseline, "held_in": held_in, "held_out": held_out, "deltas": deltas},
                "checks": deterministic_checks,
                "acceptance": {
                    "locked_surfaces_unchanged": locked_unchanged,
                    "no_regression": no_regression,
                    "strict_improvement": strict_improvement,
                    "checks_passed": checks_passed,
                    "accepted_by_gate": accepted,
                },
                "updated_at": datetime.now().isoformat(),
            }
        )
        with self._lock:
            self._write_candidate(candidate)
            self._append_event("candidate_evaluated", candidate_id, candidate["acceptance"])
            if not accepted:
                self._append_jsonl(self.rejected_file, {"candidate_id": candidate_id, "at": candidate["updated_at"], "acceptance": candidate["acceptance"], "deltas": deltas})
        return candidate

    def review_candidate(self, candidate_id: str, decision: str, reviewer: str, comment: str = "") -> Dict[str, Any]:
        if decision not in {"approve", "reject"}:
            raise ValueError("decision 必须是 approve 或 reject")
        candidate = self.get_candidate(candidate_id)
        if not candidate:
            raise KeyError(candidate_id)
        if decision == "approve" and candidate.get("status") != "awaiting_human_review":
            raise ValueError("候选尚未通过双数据集门禁，不能批准")
        candidate["status"] = "approved" if decision == "approve" else "rejected"
        candidate["decision"] = {
            "decision": decision,
            "reviewer": reviewer or "Human Reviewer",
            "comment": comment,
            "at": datetime.now().isoformat(),
        }
        candidate["updated_at"] = candidate["decision"]["at"]
        with self._lock:
            self._write_candidate(candidate)
            self._append_event("human_review", candidate_id, candidate["decision"])
            if decision == "reject":
                self._append_jsonl(self.rejected_file, {"candidate_id": candidate_id, **candidate["decision"]})
        return candidate

    def archive_candidate(self, candidate_id: str) -> bool:
        path = self._candidate_path(candidate_id)
        if not path.exists():
            return False
        with self._lock:
            record = json.loads(path.read_text(encoding="utf-8"))
            record["status"] = "archived"
            record["updated_at"] = datetime.now().isoformat()
            target = self.archive_dir / path.name
            target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            path.unlink()
            try:
                recoverable_at = str(target.relative_to(self.project_root))
            except ValueError:
                recoverable_at = str(target)
            self._append_event("candidate_archived", candidate_id, {"recoverable_at": recoverable_at})
        return True

    def get_candidate(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        path = self._candidate_path(candidate_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def list_candidates(self, limit: int = 30) -> List[Dict[str, Any]]:
        paths = sorted(self.candidate_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True)
        records: List[Dict[str, Any]] = []
        for path in paths[: max(1, limit)]:
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return records

    def summary(self) -> Dict[str, Any]:
        candidates = self.list_candidates(limit=500)
        counts: Dict[str, int] = {}
        for item in candidates:
            status = str(item.get("status") or "unknown")
            counts[status] = counts.get(status, 0) + 1
        return {
            "surfaces": self.surfaces(),
            "candidate_count": len(candidates),
            "status_counts": counts,
            "candidates": candidates[:20],
            "event_count": self._line_count(self.events_file),
            "rejected_count": self._line_count(self.rejected_file),
        }

    def _is_editable(self, surface: str) -> bool:
        normalized = surface.replace("\\", "/").lstrip("./")
        return any(normalized == allowed.rstrip("/") or normalized.startswith(allowed) for allowed in DEFAULT_SURFACES["editable"])

    def _fingerprints(self, relative_paths: Iterable[str]) -> Dict[str, Optional[str]]:
        fingerprints: Dict[str, Optional[str]] = {}
        for relative in relative_paths:
            path = self.project_root / relative
            if path.is_file():
                fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif path.is_dir():
                digest = hashlib.sha256()
                for child in sorted(path.rglob("*")):
                    if child.is_file() and "__pycache__" not in child.parts:
                        digest.update(str(child.relative_to(path)).encode("utf-8"))
                        digest.update(child.read_bytes())
                fingerprints[relative] = digest.hexdigest()
            else:
                fingerprints[relative] = None
        return fingerprints

    def _candidate_path(self, candidate_id: str) -> Path:
        safe = "".join(ch for ch in candidate_id if ch.isalnum() or ch in {"-", "_"})
        return self.candidate_dir / f"{safe}.json"

    def _write_candidate(self, record: Dict[str, Any]) -> None:
        target = self._candidate_path(str(record["candidate_id"]))
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(target)

    def _append_event(self, event_type: str, candidate_id: str, payload: Dict[str, Any]) -> None:
        self._append_jsonl(
            self.events_file,
            {
                "event_id": f"HE-{uuid.uuid4().hex[:10].upper()}",
                "event_type": event_type,
                "candidate_id": candidate_id,
                "payload": payload,
                "at": datetime.now().isoformat(),
            },
        )

    def _append_jsonl(self, path: Path, record: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

    def _line_count(self, path: Path) -> int:
        if not path.exists():
            return 0
        return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())

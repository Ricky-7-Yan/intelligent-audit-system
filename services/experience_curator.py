"""Governed experience memory inspired by self-improving agent systems."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS


class ExperienceCurator:
    """Persist reusable lessons without allowing autonomous code mutation."""

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or (PATHS["data"] / "agent_experience")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def propose(self, task: Dict[str, Any], evaluation: Dict[str, Any]) -> Dict[str, Any]:
        weakest = sorted(evaluation.get("components", []), key=lambda item: float(item.get("score") or 0))[:3]
        issues = [
            str(issue)
            for reflection in task.get("reflections", [])
            for issue in reflection.get("issues", [])
            if str(issue).strip()
        ]
        failed_assertions = [
            assertion["label"]
            for component in weakest
            for assertion in component.get("assertions", [])
            if not assertion.get("passed")
        ]
        experience_id = f"EXP-{uuid.uuid4().hex[:10].upper()}"
        record = {
            "experience_id": experience_id,
            "status": "proposed",
            "title": self._title(task, weakest),
            "task_id": task.get("task_id"),
            "objective_pattern": self._objective_pattern(task.get("objective", "")),
            "trigger": {
                "weak_components": [item.get("id") for item in weakest],
                "issues": list(dict.fromkeys(issues + failed_assertions))[:12],
                "evaluation_run_id": evaluation.get("evaluation_run_id"),
                "episode_digest": evaluation.get("trace_binding", {}).get("episode_digest"),
            },
            "lesson": {
                "when": self._objective_pattern(task.get("objective", "")),
                "do": self._actions(weakest),
                "verify": [
                    "组件评测总分不低于当前基线",
                    "关键安全断言全部通过",
                    "留出任务集无新增回归",
                ],
                "do_not": [
                    "不得自动修改评测器、权限策略或安全门禁",
                    "不得写入原始客户证据、凭据或完整上下文",
                    "未经人工批准不得应用到新任务",
                ],
            },
            "review": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        }
        self._path(experience_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    def list(self, limit: int = 30, status: Optional[str] = None) -> List[Dict[str, Any]]:
        files = sorted(self.base_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        records: List[Dict[str, Any]] = []
        for path in files:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if status and record.get("status") != status:
                continue
            records.append(record)
            if len(records) >= max(1, min(limit, 100)):
                break
        return records

    def get(self, experience_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(experience_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def review(self, experience_id: str, decision: str, reviewer: str, comment: str = "") -> Dict[str, Any]:
        record = self.get(experience_id)
        if not record:
            raise KeyError(experience_id)
        normalized = decision.lower()
        if normalized not in {"approved", "rejected"}:
            raise ValueError("decision must be approved or rejected")
        record["status"] = normalized
        record["review"] = {
            "decision": normalized,
            "reviewer": reviewer.strip() or "human-reviewer",
            "comment": comment.strip(),
            "reviewed_at": datetime.now().isoformat(),
        }
        record["updated_at"] = datetime.now().isoformat()
        self._path(experience_id).write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return record

    def relevant(self, objective: str, limit: int = 3) -> List[Dict[str, Any]]:
        objective_terms = set(self._terms(objective))
        ranked = []
        for record in self.list(limit=100, status="approved"):
            pattern_terms = set(self._terms(record.get("objective_pattern", "")))
            overlap = len(objective_terms & pattern_terms)
            if overlap:
                ranked.append((overlap / max(len(pattern_terms), 1), record))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            {
                "experience_id": record["experience_id"],
                "status": record["status"],
                "title": record["title"],
                "lesson": record["lesson"],
                "score": round(score, 3),
            }
            for score, record in ranked[: max(1, min(limit, 10))]
        ]

    def _path(self, experience_id: str) -> Path:
        safe_id = "".join(char for char in experience_id if char.isalnum() or char in {"-", "_"})
        return self.base_dir / f"{safe_id}.json"

    def _title(self, task: Dict[str, Any], weakest: List[Dict[str, Any]]) -> str:
        names = "、".join(str(item.get("name")) for item in weakest[:2])
        return f"{str(task.get('objective') or '审计任务')[:36]}：加强{names}"

    def _objective_pattern(self, objective: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(objective or "")).strip()
        return cleaned[:160] or "通用审计任务"

    def _actions(self, weakest: List[Dict[str, Any]]) -> List[str]:
        actions = []
        for component in weakest:
            failed = [
                assertion.get("label")
                for assertion in component.get("assertions", [])
                if not assertion.get("passed")
            ]
            if failed:
                actions.append(f"在{component.get('name')}阶段优先验证：{'、'.join(failed[:3])}")
        return actions or ["沿用当前任务计划，并在每一步保留验证证据和停止条件。"]

    def _terms(self, text: str) -> List[str]:
        return [
            token.lower()
            for token in re.findall(r"[\u4e00-\u9fff]{2,}|[A-Za-z][A-Za-z0-9_-]{2,}", str(text or ""))
        ]

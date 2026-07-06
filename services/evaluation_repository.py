"""Persistent evaluation run storage and release-gate decisions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS


class EvaluationRunRepository:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or PATHS["evaluation_runs"]
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_run(self, run_type: str, payload: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        run_id = f"EV-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        metrics = self._extract_metrics(run_type, results)
        comparison = self._compare_with_baseline(run_type, metrics)
        record = {
            "run_id": run_id,
            "run_type": run_type,
            "created_at": datetime.now().isoformat(),
            "payload_summary": self._payload_summary(payload),
            "metrics": metrics,
            "comparison": comparison,
            "release_gate": self._release_gate(metrics, comparison),
            "results": results,
        }
        self._path(run_id).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        files = sorted(self.base_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        records = []
        for path in files[: max(limit, 1)]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            records.append(
                {
                    "run_id": record.get("run_id"),
                    "run_type": record.get("run_type"),
                    "created_at": record.get("created_at"),
                    "payload_summary": record.get("payload_summary", {}),
                    "metrics": record.get("metrics", {}),
                    "comparison": record.get("comparison", {}),
                    "release_gate": record.get("release_gate", {}),
                }
            )
        return records

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _path(self, run_id: str) -> Path:
        safe_id = "".join(ch for ch in run_id if ch.isalnum() or ch in {"-", "_"})
        return self.base_dir / f"{safe_id}.json"

    def _payload_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cases = payload.get("test_cases") or payload.get("cases") or []
        return {
            "model_path": payload.get("model_path", "current-agent"),
            "case_count": len(cases) if isinstance(cases, list) else 0,
            "customized": bool(cases),
        }

    def _extract_metrics(self, run_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
        if run_type == "rag":
            total = int(results.get("total_cases") or 0)
            regressions = sum(
                1
                for item in results.get("results", [])
                for mode in item.get("failure_modes", [])
                if "未发现" not in str(mode)
            )
            return {
                "overall_score": float(results.get("overall_score") or 0),
                "total_tests": total,
                "pass_rate": round(sum(1 for item in results.get("results", []) if float(item.get("overall") or 0) >= 0.7) / total, 3) if total else 0,
                "regression_count": regressions,
                "avg_latency_ms": None,
            }
        if run_type == "research":
            evaluation = results.get("evaluation", {})
            return {
                "overall_score": float(evaluation.get("faithfulness") or 0),
                "total_tests": len(results.get("query_rewrites", [])),
                "pass_rate": 0 if evaluation.get("requires_human_review") else 1,
                "regression_count": 1 if evaluation.get("requires_human_review") else 0,
                "avg_latency_ms": None,
            }
        metrics = results.get("overall_metrics", {})
        return {
            "overall_score": float(metrics.get("overall_score") or 0),
            "total_tests": int(metrics.get("total_tests") or 0),
            "pass_rate": float(metrics.get("pass_rate") or 0),
            "regression_count": int(metrics.get("regression_count") or 0),
            "avg_latency_ms": metrics.get("avg_latency_ms"),
        }

    def _compare_with_baseline(self, run_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        baseline = next((item for item in self.list_runs(limit=100) if item.get("run_type") == run_type), None)
        if not baseline:
            return {"baseline_run_id": None, "deltas": {}, "regressions": [], "status": "baseline_created"}
        previous = baseline.get("metrics", {})
        deltas: Dict[str, Any] = {}
        regressions = []
        for key in ("overall_score", "pass_rate"):
            current_value = float(metrics.get(key) or 0)
            previous_value = float(previous.get(key) or 0)
            delta = round(current_value - previous_value, 4)
            deltas[key] = delta
            if previous_value and delta < -0.05:
                regressions.append(f"{key} 较基线下降 {abs(delta):.1%}")
        current_latency = metrics.get("avg_latency_ms")
        previous_latency = previous.get("avg_latency_ms")
        if isinstance(current_latency, (int, float)) and isinstance(previous_latency, (int, float)):
            deltas["avg_latency_ms"] = round(float(current_latency) - float(previous_latency), 2)
            if previous_latency and current_latency > previous_latency * 1.25:
                regressions.append("平均延迟较基线上升超过 25%")
        return {
            "baseline_run_id": baseline.get("run_id"),
            "deltas": deltas,
            "regressions": regressions,
            "status": "regression" if regressions else "stable",
        }

    def _release_gate(self, metrics: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        score = float(metrics.get("overall_score") or 0)
        pass_rate = float(metrics.get("pass_rate") or 0)
        regressions = int(metrics.get("regression_count") or 0)
        blockers = []
        if score < 0.75:
            blockers.append("整体得分低于 0.75。")
        if pass_rate < 0.7:
            blockers.append("通过率低于 70%。")
        if regressions > 0:
            blockers.append("存在需要处理的回归风险。")
        if comparison and comparison.get("regressions"):
            blockers.extend(comparison["regressions"])
        if not blockers:
            status = "pass"
            label = "可发布"
        elif score >= 0.6 and regressions <= 2:
            status = "review"
            label = "需复核"
        else:
            status = "blocked"
            label = "阻断发布"
        return {
            "status": status,
            "label": label,
            "blockers": blockers,
            "baseline_run_id": (comparison or {}).get("baseline_run_id"),
            "deltas": (comparison or {}).get("deltas", {}),
            "thresholds": {"overall_score": 0.75, "pass_rate": 0.7, "regression_count": 0},
        }

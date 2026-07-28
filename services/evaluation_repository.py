"""Persistent evaluation run storage and release-gate decisions."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS
from services.evaluation_calibration import beta_posterior_mean, wilson_lower_bound
from services.security import current_tenant_id, record_visible
from services.record_store import SQLiteRecordStore


class EvaluationRunRepository:
    def __init__(self, base_dir: Path | None = None) -> None:
        self.base_dir = base_dir or PATHS["evaluation_runs"]
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.store = SQLiteRecordStore(self.base_dir / ".records.sqlite3")
        self._migrate_legacy_records()

    def create_run(self, run_type: str, payload: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
        run_id = f"EV-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}"
        metrics = self._extract_metrics(run_type, results)
        comparison = self._compare_with_baseline(run_type, metrics)
        record = {
            "run_id": run_id,
            "tenant_id": current_tenant_id(),
            "run_type": run_type,
            "created_at": datetime.now().isoformat(),
            "payload_summary": self._payload_summary(payload),
            "metrics": metrics,
            "comparison": comparison,
            "release_gate": self._release_gate(metrics, comparison),
            "results": results,
        }
        self._write_record(record)
        return record

    def list_runs(self, limit: int = 20) -> List[Dict[str, Any]]:
        records = []
        for record in self.store.list("evaluation_run", limit=max(limit, 1)):
            if not record_visible(record):
                continue
            records.append(
                {
                    "run_id": record.get("run_id"),
                    "run_type": record.get("run_type"),
                    "created_at": record.get("created_at"),
                    "is_baseline": bool(record.get("is_baseline")),
                    "payload_summary": record.get("payload_summary", {}),
                    "metrics": record.get("metrics", {}),
                    "comparison": record.get("comparison", {}),
                    "release_gate": record.get("release_gate", {}),
                }
            )
        return records

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        record = self.store.get("evaluation_run", run_id)
        if record is None:
            path = self._path(run_id)
            if path.exists():
                try:
                    record = json.loads(path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    return None
                if record_visible(record):
                    record.setdefault("tenant_id", current_tenant_id())
                    self.store.put("evaluation_run", run_id, record)
        return record if record is not None and record_visible(record) else None

    def delete_run(self, run_id: str) -> bool:
        path = self._path(run_id)
        if self.get_run(run_id) is None:
            return False
        removed = self.store.delete("evaluation_run", run_id)
        if path.exists():
            path.unlink()
        return removed

    def _path(self, run_id: str) -> Path:
        safe_id = "".join(ch for ch in run_id if ch.isalnum() or ch in {"-", "_"})
        return self.base_dir / f"{safe_id}.json"

    def _payload_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        cases = payload.get("test_cases") or payload.get("cases") or []
        return {
            "model_path": payload.get("model_path", "current-agent"),
            "case_count": int(payload.get("case_count") or (len(cases) if isinstance(cases, list) else 0)),
            "customized": bool(cases),
            "task_id": payload.get("task_id"),
            "source": payload.get("source"),
        }

    def _extract_metrics(self, run_type: str, results: Dict[str, Any]) -> Dict[str, Any]:
        if run_type in {"task_component", "runtime_component"}:
            summary = results.get("summary", {})
            critical_failures = summary.get("critical_failures") or []
            return {
                "overall_score": float(summary.get("overall_score") or 0),
                "total_tests": int(summary.get("total_tests") or summary.get("component_count") or 0),
                "pass_rate": float(summary.get("pass_rate") or 0),
                "raw_pass_rate": float(summary.get("raw_pass_rate") or 0),
                "confidence_lower_bound": float(summary.get("confidence_lower_bound") or 0),
                "evidence_coverage": float(summary.get("evidence_coverage") or 0),
                "trial_count": int(summary.get("trial_count") or summary.get("total_tests") or 1),
                "regression_count": len(critical_failures),
                "critical_failures": critical_failures,
                "avg_latency_ms": summary.get("avg_latency_ms"),
                "required_score": 0.82,
                "required_pass_rate": 0.72,
                "required_confidence_lower_bound": 0.68,
                "required_evidence_coverage": 0.72,
            }
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
                "pass_rate": beta_posterior_mean(
                    sum(1 for item in results.get("results", []) if float(item.get("overall") or 0) >= 0.7),
                    total,
                ) if total else 0,
                "confidence_lower_bound": wilson_lower_bound(
                    sum(1 for item in results.get("results", []) if float(item.get("overall") or 0) >= 0.7),
                    total,
                ),
                "trial_count": total,
                "regression_count": regressions,
                "avg_latency_ms": None,
            }
        if run_type == "research":
            evaluation = results.get("evaluation", {})
            return {
                "overall_score": float(evaluation.get("faithfulness") or 0),
                "total_tests": len(results.get("query_rewrites", [])),
                "pass_rate": beta_posterior_mean(
                    0 if evaluation.get("requires_human_review") else 1,
                    1,
                ),
                "confidence_lower_bound": wilson_lower_bound(
                    0 if evaluation.get("requires_human_review") else 1,
                    1,
                ),
                "trial_count": 1,
                "regression_count": 1 if evaluation.get("requires_human_review") else 0,
                "avg_latency_ms": None,
            }
        metrics = results.get("overall_metrics", {})
        return {
            "overall_score": float(metrics.get("overall_score") or 0),
            "total_tests": int(metrics.get("total_tests") or 0),
            "pass_rate": float(metrics.get("pass_rate") or 0),
            "confidence_lower_bound": float(metrics.get("confidence_lower_bound") or 0),
            "trial_count": int(metrics.get("total_tests") or 0),
            "regression_count": int(metrics.get("regression_count") or 0),
            "avg_latency_ms": metrics.get("avg_latency_ms"),
        }

    def _compare_with_baseline(self, run_type: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        same_type_runs = [item for item in self.list_runs(limit=100) if item.get("run_type") == run_type]
        baseline = next((item for item in same_type_runs if item.get("is_baseline")), None)
        if baseline is None:
            baseline = same_type_runs[0] if same_type_runs else None
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
            "baseline_locked": bool(baseline.get("is_baseline")),
            "deltas": deltas,
            "regressions": regressions,
            "status": "regression" if regressions else "stable",
        }

    def mark_as_baseline(self, run_id: str) -> Optional[Dict[str, Any]]:
        record = self.get_run(run_id)
        if not record:
            return None
        for item in self.list_runs(limit=200):
            if item.get("run_type") != record.get("run_type"):
                continue
            current = self.get_run(item["run_id"]) or item
            current["is_baseline"] = False
            self._write_record(current)
        updated = self.get_run(run_id) or record
        updated["is_baseline"] = True
        updated["updated_at"] = datetime.now().isoformat()
        self._write_record(updated)
        return updated

    def _write_record(self, record: Dict[str, Any]) -> None:
        record.setdefault("tenant_id", current_tenant_id())
        expected = record.get("_storage_version")
        version = self.store.put(
            "evaluation_run",
            str(record["run_id"]),
            record,
            expected_version=int(expected) if expected is not None else None,
        )
        record["_storage_version"] = version

    def _migrate_legacy_records(self) -> None:
        for path in self.base_dir.glob("*.json"):
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
                run_id = str(record.get("run_id") or path.stem)
                if self.store.get("evaluation_run", run_id) is None and record_visible(record):
                    record.setdefault("tenant_id", current_tenant_id())
                    self.store.put("evaluation_run", run_id, record)
            except (OSError, json.JSONDecodeError):
                continue

    def _release_gate(self, metrics: Dict[str, Any], comparison: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        score = float(metrics.get("overall_score") or 0)
        pass_rate = float(metrics.get("pass_rate") or 0)
        regressions = int(metrics.get("regression_count") or 0)
        blockers = []
        required_score = float(metrics.get("required_score") or 0.75)
        required_pass_rate = float(metrics.get("required_pass_rate") or 0.70)
        required_lower_bound = float(metrics.get("required_confidence_lower_bound") or 0.60)
        required_evidence_coverage = float(metrics.get("required_evidence_coverage") or 0.0)
        confidence_lower_bound = float(metrics.get("confidence_lower_bound") or 0)
        evidence_coverage = float(metrics.get("evidence_coverage") or 0)
        trial_count = int(metrics.get("trial_count") or metrics.get("total_tests") or 0)
        review_reasons = []
        if score < required_score:
            blockers.append(f"整体得分低于 {required_score:.2f}。")
        if pass_rate < required_pass_rate:
            blockers.append(f"通过率低于 {required_pass_rate:.0%}。")
        if required_evidence_coverage and evidence_coverage < required_evidence_coverage:
            blockers.append(f"证据覆盖率低于 {required_evidence_coverage:.0%}。")
        if confidence_lower_bound < required_lower_bound:
            review_reasons.append(f"95% 置信下界低于 {required_lower_bound:.0%}。")
        if trial_count < 3:
            review_reasons.append("独立试验少于 3 次，不能自动判定为可发布。")
        if regressions > 0:
            blockers.append("存在需要处理的回归风险。")
        blockers.extend(str(item) for item in metrics.get("critical_failures", []) if str(item).strip())
        if comparison and comparison.get("regressions"):
            blockers.extend(comparison["regressions"])
        if blockers:
            status = "blocked"
            label = "阻断发布"
        elif review_reasons:
            status = "review"
            label = "需复核"
        else:
            status = "pass"
            label = "可发布"
        return {
            "status": status,
            "label": label,
            "blockers": blockers,
            "review_reasons": review_reasons,
            "baseline_run_id": (comparison or {}).get("baseline_run_id"),
            "deltas": (comparison or {}).get("deltas", {}),
            "thresholds": {
                "overall_score": required_score,
                "pass_rate": required_pass_rate,
                "confidence_lower_bound": required_lower_bound,
                "evidence_coverage": required_evidence_coverage,
                "minimum_trials": 3,
                "regression_count": 0,
            },
        }

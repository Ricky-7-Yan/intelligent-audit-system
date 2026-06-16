"""Evidence file analysis for audit fieldwork."""

from __future__ import annotations

import csv
import json
import re
import uuid
from collections import Counter
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List

from agents.audit_agent import CONTROL_LIBRARY
from config import PATHS


RISK_PATTERNS = [
    ("privileged_access", "特权权限", ["admin", "administrator", "root", "superuser", "特权", "管理员"], "高"),
    ("terminated_user", "离职账号", ["terminated", "inactive", "离职", "停用", "禁用"], "高"),
    ("sod_conflict", "职责分离冲突", ["制单", "审批", "复核", "付款", "posting", "approve"], "高"),
    ("change_without_approval", "变更审批缺失", ["未经审批", "no approval", "missing approval", "紧急变更"], "中"),
    ("failed_login", "异常登录", ["failed", "失败", "拒绝", "denied", "locked"], "中"),
    ("sensitive_data", "敏感数据", ["身份证", "手机号", "银行卡", "salary", "password", "secret"], "中"),
]


class EvidenceAnalyzer:
    def __init__(self, root: Path | None = None) -> None:
        self.root = root or PATHS["evidence_analyses"]
        self.root.mkdir(parents=True, exist_ok=True)

    def analyze_file(self, file_name: str, content: bytes, audit_context: Dict[str, Any] | None = None) -> Dict[str, Any]:
        audit_context = audit_context or {}
        suffix = Path(file_name or "").suffix.lower()
        text = self._decode(content)
        rows = self._parse_rows(text, suffix)
        profile = self._profile(file_name, suffix, text, rows)
        signals = self._risk_signals(text, rows)
        mapped_controls = self._map_controls(text, rows, audit_context)
        requests = self._evidence_requests(profile, signals, mapped_controls)
        result = {
            "analysis_id": f"EA-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6].upper()}",
            "file_name": file_name,
            "created_at": datetime.now().isoformat(),
            "audit_context": audit_context,
            "profile": profile,
            "risk_signals": signals,
            "mapped_controls": mapped_controls,
            "evidence_requests": requests,
            "quality_gate": self._quality_gate(profile, signals, mapped_controls),
            "recommended_next_steps": self._next_steps(signals, mapped_controls),
        }
        self._path(result["analysis_id"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def list_analyses(self, limit: int = 20) -> List[Dict[str, Any]]:
        files = sorted(self.root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        items = []
        for path in files[: max(limit, 1)]:
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            items.append(
                {
                    "analysis_id": record.get("analysis_id"),
                    "file_name": record.get("file_name"),
                    "created_at": record.get("created_at"),
                    "profile": record.get("profile", {}),
                    "quality_gate": record.get("quality_gate", {}),
                    "risk_count": len(record.get("risk_signals", [])),
                    "control_count": len(record.get("mapped_controls", [])),
                }
            )
        return items

    def get_analysis(self, analysis_id: str) -> Dict[str, Any] | None:
        path = self._path(analysis_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

    def _decode(self, content: bytes) -> str:
        for encoding in ["utf-8-sig", "utf-8", "gb18030", "gbk", "latin-1"]:
            try:
                return content.decode(encoding)
            except UnicodeDecodeError:
                continue
        return content.decode("utf-8", errors="replace")

    def _parse_rows(self, text: str, suffix: str) -> List[Dict[str, Any]]:
        if suffix == ".json":
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return []
            if isinstance(data, list):
                return [item for item in data if isinstance(item, dict)]
            if isinstance(data, dict):
                for value in data.values():
                    if isinstance(value, list):
                        return [item for item in value if isinstance(item, dict)]
                return [data]
            return []
        if suffix in {".csv", ".tsv"}:
            delimiter = "\t" if suffix == ".tsv" else ","
            try:
                return list(csv.DictReader(StringIO(text), delimiter=delimiter))[:5000]
            except csv.Error:
                return []
        if "," in text.splitlines()[0] if text.splitlines() else False:
            try:
                return list(csv.DictReader(StringIO(text)))[:5000]
            except csv.Error:
                return []
        return []

    def _profile(self, file_name: str, suffix: str, text: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        fields = list(rows[0].keys()) if rows else []
        non_empty = Counter()
        distinct = {}
        for row in rows[:1000]:
            for field, value in row.items():
                if str(value or "").strip():
                    non_empty[field] += 1
        for field in fields[:40]:
            distinct[field] = len({str(row.get(field, "")).strip() for row in rows[:1000] if str(row.get(field, "")).strip()})
        return {
            "file_type": suffix.lstrip(".") or "text",
            "file_name": file_name,
            "rows": len(rows),
            "characters": len(text),
            "fields": fields[:80],
            "field_count": len(fields),
            "non_empty_rate": {field: round(non_empty[field] / max(len(rows), 1), 3) for field in fields[:40]},
            "distinct_values": distinct,
            "sample_rows": rows[:5],
        }

    def _risk_signals(self, text: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        lower_text = text.lower()
        row_texts = [" ".join(str(value) for value in row.values()).lower() for row in rows[:5000]]
        signals = []
        for signal_id, title, keywords, severity in RISK_PATTERNS:
            text_hits = sum(lower_text.count(keyword.lower()) for keyword in keywords)
            row_hits = sum(1 for row_text in row_texts if any(keyword.lower() in row_text for keyword in keywords))
            if text_hits or row_hits:
                signals.append(
                    {
                        "signal_id": signal_id,
                        "title": title,
                        "severity": severity,
                        "keyword_hits": text_hits,
                        "affected_rows": row_hits,
                        "confidence": round(min(0.35 + (row_hits or text_hits) / 20, 0.95), 2),
                        "audit_implication": self._implication(signal_id),
                    }
                )
        return sorted(signals, key=lambda item: (item["severity"] != "高", -item["confidence"]))

    def _map_controls(self, text: str, rows: List[Dict[str, Any]], audit_context: Dict[str, Any]) -> List[Dict[str, Any]]:
        corpus = f"{text[:20000]} {' '.join(str(v) for row in rows[:100] for v in row.values())} {json.dumps(audit_context, ensure_ascii=False)}".lower()
        mapped = []
        for control in CONTROL_LIBRARY:
            hit_keywords = [keyword for keyword in control.get("keywords", []) if keyword.lower() in corpus]
            if not hit_keywords:
                continue
            mapped.append(
                {
                    "control_id": control["id"],
                    "domain": control["domain"],
                    "hit_keywords": hit_keywords,
                    "objective": control["objective"],
                    "test_procedure": control["test_procedure"],
                    "evidence_required": control["evidence_required"],
                }
            )
        return mapped[:8]

    def _evidence_requests(self, profile: Dict[str, Any], signals: List[Dict[str, Any]], controls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        requests = []
        for control in controls[:5]:
            requests.append(
                {
                    "evidence": f"{control['domain']} 支撑底稿",
                    "usage": f"验证 {control['control_id']} 的设计和运行有效性",
                    "required_fields": control.get("evidence_required", [])[:4],
                    "priority": "高" if any(signal["severity"] == "高" for signal in signals) else "中",
                }
            )
        if profile["rows"] and any(signal["signal_id"] == "terminated_user" for signal in signals):
            requests.append(
                {
                    "evidence": "HR 离职清单与账号禁用记录交叉核对",
                    "usage": "验证离职账号是否及时禁用",
                    "required_fields": ["员工编号", "离职日期", "账号状态", "禁用时间"],
                    "priority": "高",
                }
            )
        return requests[:8]

    def _quality_gate(self, profile: Dict[str, Any], signals: List[Dict[str, Any]], controls: List[Dict[str, Any]]) -> Dict[str, Any]:
        completeness = 0.35
        if profile["rows"] > 0:
            completeness += 0.25
        if profile["field_count"] >= 4:
            completeness += 0.15
        if controls:
            completeness += 0.15
        if signals:
            completeness += 0.1
        confidence = round(min(completeness, 0.95), 2)
        return {
            "status": "pass" if confidence >= 0.75 else "review",
            "confidence": confidence,
            "review_required": any(signal["severity"] == "高" for signal in signals) or confidence < 0.75,
            "note": "已形成可复核的证据画像。" if confidence >= 0.75 else "证据结构或控制映射不足，建议补充字段说明和来源背景。",
        }

    def _next_steps(self, signals: List[Dict[str, Any]], controls: List[Dict[str, Any]]) -> List[str]:
        steps = ["将该证据加入当前审计项目底稿索引。"]
        if signals:
            steps.append("对高风险信号执行抽样复核，并记录例外处理结论。")
        if controls:
            steps.append("将映射控制加入控制测试工作台，补充测试人、样本范围和例外说明。")
        steps.append("把字段口径、导出时间和系统来源写入证据元数据。")
        return steps

    def _implication(self, signal_id: str) -> str:
        return {
            "privileged_access": "需要复核特权账号审批、使用日志和定期复核记录。",
            "terminated_user": "可能存在账号生命周期控制缺陷，需要与 HR 离职数据交叉验证。",
            "sod_conflict": "可能存在职责分离冲突，需要检查补偿性控制和例外审批。",
            "change_without_approval": "需要补充变更单、审批链、测试记录和上线授权。",
            "failed_login": "需要检查异常登录监控、锁定策略和告警处置记录。",
            "sensitive_data": "需要确认敏感数据分类分级、访问授权、加密脱敏和共享审批。",
        }.get(signal_id, "需要人工复核该风险信号。")

    def _path(self, analysis_id: str) -> Path:
        safe_id = "".join(ch for ch in analysis_id if ch.isalnum() or ch in {"-", "_"})
        return self.root / f"{safe_id}.json"

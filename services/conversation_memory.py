"""Transactional working, episodic, and profile memory for audit conversations."""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PATHS
from services.record_store import SQLiteRecordStore
from services.security import current_tenant_id, record_visible


class ConversationMemory:
    """Persist conversation context with tenant-scoped transactional updates."""

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        compress_at: int = 18,
        retain_recent: int = 8,
        context_budget_tokens: int = 2400,
    ) -> None:
        self.base_dir = base_dir or PATHS["data"] / "conversation_memory"
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.compress_at = compress_at
        self.retain_recent = retain_recent
        self.context_budget_tokens = max(600, context_budget_tokens)
        self._lock = threading.RLock()
        self.store = SQLiteRecordStore(self.base_dir / ".records.sqlite3")

    def context_for(self, session_id: str, message: str) -> Dict[str, Any]:
        session = self.get_session(session_id) or self._empty(session_id)
        related = self._related(session.get("messages", []), message)
        prompt_parts = []
        if session.get("summary"):
            prompt_parts.append(f"[会话摘要]\n{session['summary']}")
        if session.get("profile"):
            prompt_parts.append(f"[审计画像]\n{json.dumps(session['profile'], ensure_ascii=False)}")
        if related:
            prompt_parts.append("[相关历史]\n" + "\n".join(f"- {item['content'][:240]}" for item in related))
        recent = session.get("messages", [])[-6:]
        if recent:
            prompt_parts.append(
                "[最近对话]\n" + "\n".join(f"{item['role']}: {item['content'][:360]}" for item in recent)
            )
        prompt_text = self._fit_context(prompt_parts)
        estimated_tokens = self._estimate_tokens(prompt_text)
        return {
            "session_id": session_id,
            "summary": session.get("summary", ""),
            "profile": session.get("profile", {}),
            "related_messages": related,
            "recent_messages": recent,
            "prompt_text": prompt_text,
            "context_budget": {
                "limit_tokens": self.context_budget_tokens,
                "estimated_tokens": estimated_tokens,
                "utilization": round(estimated_tokens / max(self.context_budget_tokens, 1), 3),
                "policy": "JIT retrieval → observation masking → compaction",
                "cache_stable_prefix": True,
            },
            "memory_layers": {
                "working": len(session.get("messages", [])),
                "episodic": len(session.get("episodes", [])),
                "profile_fields": len(session.get("profile", {})),
            },
        }

    def record_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            session = self.get_session(session_id) or self._empty(session_id)
            now = datetime.now().isoformat()
            session["messages"].extend(
                [
                    {"role": "user", "content": user_message, "at": now},
                    {"role": "assistant", "content": assistant_message, "at": now},
                ]
            )
            session["profile"] = self._update_profile(session.get("profile", {}), user_message, metadata or {})
            session["turns"] = int(session.get("turns", 0)) + 1
            session["updated_at"] = now
            if len(session["messages"]) >= self.compress_at:
                self._compress(session)
            self._write(session)
            return self._summary(session)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        session = self.store.get("conversation_session", session_id)
        if session is not None:
            return session
        path = self._path(session_id)
        if not path.exists():
            return None
        try:
            session = json.loads(path.read_text(encoding="utf-8"))
            if not record_visible(session):
                return None
            session.setdefault("tenant_id", current_tenant_id())
            self._write(session)
            return session
        except (OSError, json.JSONDecodeError):
            return None

    def list_sessions(self, limit: int = 20) -> List[Dict[str, Any]]:
        self._migrate_legacy_sessions()
        records = self.store.list("conversation_session", limit=max(limit, 1))
        return [self._summary(session) for session in records]

    def delete_session(self, session_id: str) -> bool:
        """Delete a user-manageable conversation record."""
        path = self._path(session_id)
        if self.get_session(session_id) is None:
            return False
        removed = self.store.delete("conversation_session", session_id)
        if path.exists():
            path.unlink()
        return removed

    def stats(self) -> Dict[str, Any]:
        sessions = self.list_sessions(limit=1000)
        return {
            "sessions": len(sessions),
            "turns": sum(int(item.get("turns", 0)) for item in sessions),
            "working_messages": sum(int(item.get("working_messages", 0)) for item in sessions),
            "episodes": sum(int(item.get("episodes", 0)) for item in sessions),
        }

    def _empty(self, session_id: str) -> Dict[str, Any]:
        now = datetime.now().isoformat()
        return {
            "session_id": session_id,
            "tenant_id": current_tenant_id(),
            "summary": "",
            "profile": {},
            "messages": [],
            "episodes": [],
            "turns": 0,
            "created_at": now,
            "updated_at": now,
        }

    def _compress(self, session: Dict[str, Any]) -> None:
        archived = session["messages"][:-self.retain_recent]
        if not archived:
            return
        user_points = [item["content"] for item in archived if item.get("role") == "user"][-5:]
        assistant_points = [item["content"] for item in archived if item.get("role") == "assistant"][-3:]
        summary = "；".join([*user_points, *assistant_points])
        summary = re.sub(r"\s+", " ", summary)[:1400]
        previous = session.get("summary", "")
        session["summary"] = (f"{previous}；{summary}" if previous else summary)[-2200:]
        session.setdefault("episodes", []).append(
            {
                "episode_id": f"EP-{len(session.get('episodes', [])) + 1:04d}",
                "summary": summary,
                "message_count": len(archived),
                "confidence": 0.72,
                "status": "active",
                "valid_from": datetime.now().isoformat(),
                "valid_until": None,
                "privacy": "session",
                "helpful_count": 0,
                "harmful_count": 0,
                "created_at": datetime.now().isoformat(),
            }
        )
        session["episodes"] = session["episodes"][-20:]
        session["messages"] = session["messages"][-self.retain_recent:]

    def _update_profile(self, profile: Dict[str, Any], message: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(profile)
        standards = set(updated.get("standards", []))
        for standard in ("ISO27001", "SOX", "COBIT", "数据安全法"):
            if standard.lower() in message.lower():
                standards.add(standard)
        topics = set(updated.get("risk_topics", []))
        for topic in ("权限", "账号", "变更", "日志", "数据", "备份", "接口", "财务", "供应商"):
            if topic in message:
                topics.add(topic)
        systems = set(updated.get("systems", []))
        systems.update(
            re.findall(r"[\w\u4e00-\u9fff-]{1,24}(?:ERP|CRM|OA|系统|平台|数据库)", message, flags=re.IGNORECASE)
        )
        updated.update(
            {
                "standards": sorted(standards),
                "risk_topics": sorted(topics),
                "systems": sorted(systems)[:12],
                "last_intent": metadata.get("intent") or updated.get("last_intent"),
                "last_agents": metadata.get("agents") or updated.get("last_agents", []),
                "updated_at": datetime.now().isoformat(),
            }
        )
        return updated

    def _related(self, messages: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        query_tokens = self._tokens(query)
        ranked = []
        for item in messages[:-2]:
            tokens = self._tokens(str(item.get("content", "")))
            if not query_tokens or not tokens:
                continue
            score = len(query_tokens & tokens) / max(len(query_tokens | tokens), 1)
            if score >= 0.08:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: pair[0], reverse=True)
        return [{**item, "score": round(score, 3)} for score, item in ranked[:3]]

    def _tokens(self, text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]{1,4}", text.lower()))

    def _fit_context(self, parts: List[str]) -> str:
        """Keep high-signal sections within budget without mutating durable memory."""
        if not parts:
            return ""
        selected: List[str] = []
        used_tokens = 0
        mask = "\n[其余内容已遮罩，可从会话记忆按需取回]"
        for part in parts:
            separator_tokens = self._estimate_tokens("\n\n") if selected else 0
            remaining = self.context_budget_tokens - used_tokens - separator_tokens
            if remaining <= 0:
                break
            part_tokens = self._estimate_tokens(part)
            if part_tokens <= remaining:
                selected.append(part)
                used_tokens += separator_tokens + part_tokens
                continue

            mask_tokens = self._estimate_tokens(mask)
            available = max(0, remaining - mask_tokens)
            low, high = 0, len(part)
            while low < high:
                middle = (low + high + 1) // 2
                if self._estimate_tokens(part[:middle]) <= available:
                    low = middle
                else:
                    high = middle - 1
            reference = f"{part[:low]}{mask}" if low else mask.strip()
            if self._estimate_tokens(reference) <= remaining:
                selected.append(reference)
            break
        return "\n\n".join(selected)

    def _estimate_tokens(self, text: str) -> int:
        if not text:
            return 0
        latin = len(re.findall(r"[A-Za-z0-9_]+", text))
        cjk = len(re.findall(r"[\u4e00-\u9fff]", text))
        symbols = max(0, len(text) - cjk)
        return max(1, int(cjk * 0.9 + latin * 0.35 + symbols * 0.18))

    def _summary(self, session: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "session_id": session.get("session_id"),
            "turns": session.get("turns", 0),
            "working_messages": len(session.get("messages", [])),
            "episodes": len(session.get("episodes", [])),
            "profile": session.get("profile", {}),
            "summary": session.get("summary", ""),
            "updated_at": session.get("updated_at"),
        }

    def _path(self, session_id: str) -> Path:
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", session_id)[:120] or "session"
        return self.base_dir / f"{safe_id}.json"

    def _write(self, session: Dict[str, Any]) -> None:
        session.setdefault("tenant_id", current_tenant_id())
        expected = session.get("_storage_version")
        version = self.store.put(
            "conversation_session",
            str(session["session_id"]),
            session,
            expected_version=int(expected) if expected is not None else None,
        )
        session["_storage_version"] = version

    def _migrate_legacy_sessions(self) -> None:
        for path in self.base_dir.glob("*.json"):
            try:
                session = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            session_id = str(session.get("session_id") or path.stem)
            if record_visible(session) and self.store.get("conversation_session", session_id) is None:
                session.setdefault("tenant_id", current_tenant_id())
                self.store.put("conversation_session", session_id, session)

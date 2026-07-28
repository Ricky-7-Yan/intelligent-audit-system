"""Transactional multi-tenant JSON record storage backed by SQLite."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from services.security import current_tenant_id


class TenantBoundaryError(PermissionError):
    """Raised when a caller attempts to write into another tenant."""


class RecordConflictError(RuntimeError):
    """Raised when optimistic-locking detects a stale record update."""


class SQLiteRecordStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._initialize()

    def put(
        self,
        namespace: str,
        record_id: str,
        payload: Dict[str, Any],
        *,
        expected_version: Optional[int] = None,
    ) -> int:
        tenant_id = current_tenant_id()
        claimed_tenant = str(payload.get("tenant_id") or tenant_id)
        if claimed_tenant != tenant_id:
            raise TenantBoundaryError(
                f"cross-tenant write denied: active={tenant_id!r}, claimed={claimed_tenant!r}"
            )
        now = datetime.now(timezone.utc).isoformat()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT version FROM records WHERE namespace = ? AND tenant_id = ? AND record_id = ?",
                (namespace, tenant_id, record_id),
            ).fetchone()
            current_version = int(row[0]) if row else 0
            if expected_version is not None and current_version != int(expected_version):
                connection.rollback()
                raise RecordConflictError(
                    f"stale record {namespace}/{record_id}: expected version "
                    f"{expected_version}, current version {current_version}"
                )
            next_version = current_version + 1
            stored_payload = {
                **payload,
                "tenant_id": tenant_id,
                "_storage_version": next_version,
            }
            serialized = json.dumps(stored_payload, ensure_ascii=False, sort_keys=True, default=str)
            if row:
                connection.execute(
                    """
                    UPDATE records
                    SET payload = ?, updated_at = ?, version = ?
                    WHERE namespace = ? AND tenant_id = ? AND record_id = ?
                    """,
                    (serialized, now, next_version, namespace, tenant_id, record_id),
                )
            else:
                connection.execute(
                    """
                    INSERT INTO records(namespace, tenant_id, record_id, payload, created_at, updated_at, version)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (namespace, tenant_id, record_id, serialized, now, now, next_version),
                )
            connection.commit()
            return next_version

    def get(self, namespace: str, record_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM records WHERE namespace = ? AND tenant_id = ? AND record_id = ?",
                (namespace, current_tenant_id(), record_id),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def list(self, namespace: str, limit: int = 100) -> List[Dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT payload FROM records
                WHERE namespace = ? AND tenant_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (namespace, current_tenant_id(), max(1, int(limit))),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def delete(self, namespace: str, record_id: str) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                "DELETE FROM records WHERE namespace = ? AND tenant_id = ? AND record_id = ?",
                (namespace, current_tenant_id(), record_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def health(self) -> Dict[str, Any]:
        try:
            with self._connect() as connection:
                mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
                quick_check = connection.execute("PRAGMA quick_check").fetchone()[0]
                schema_version = connection.execute("PRAGMA user_version").fetchone()[0]
                count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
            return {
                "ready": mode.lower() == "wal" and quick_check == "ok",
                "journal_mode": mode,
                "quick_check": quick_check,
                "schema_version": schema_version,
                "records": count,
                "path": str(self.path),
            }
        except sqlite3.Error as exc:
            return {"ready": False, "error": str(exc), "path": str(self.path)}

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA user_version=1")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS records(
                    namespace TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    record_id TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    version INTEGER NOT NULL DEFAULT 1,
                    PRIMARY KEY(namespace, tenant_id, record_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_records_listing ON records(namespace, tenant_id, updated_at DESC)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            yield connection
        finally:
            connection.close()

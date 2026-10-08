"""Append-only audit trail for security-relevant decisions (denials, approvals, injections, tools)."""
from __future__ import annotations

import json
import time

from app.db import Database
from app.logging_config import get_logger

log = get_logger("audit")


class AuditLog:
    def __init__(self, db: Database):
        self.db = db

    async def record(self, *, user_id: str | None, role: str | None, action: str, outcome: str,
                     resource: str = "", **detail) -> None:
        log.info("audit", user_id=user_id, role=role, action=action, resource=resource, outcome=outcome)
        try:
            await self.db.execute(
                "INSERT INTO audit(ts,user_id,role,action,resource,outcome,detail) VALUES (?,?,?,?,?,?,?)",
                (time.time(), user_id, role, action, resource, outcome, json.dumps(detail, default=str)[:2000]),
            )
        except Exception as exc:  # auditing must never break the request path
            log.error("audit_write_failed", error=str(exc))

    async def recent(self, limit: int = 100) -> list[dict]:
        return await self.db.fetchall("SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))

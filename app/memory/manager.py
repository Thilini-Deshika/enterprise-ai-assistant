"""Conversation memory.

Three tiers:
  1. Working memory  - last N turns verbatim, in-process per session (survives all turns of a session).
  2. Session summary - older turns are compressed by the LLM into a rolling summary (token budget control).
  3. Long-term memory - SQLite: per-user *episodes* (question -> short answer + cited docs) and explicit
     *preferences* ("remember that I prefer ..."). Recalled by keyword overlap x recency x feedback weight.
Safety properties: sessions are owned by their creator; long-term items carry the access level of the
evidence they summarise and are filtered by the *current* role at recall time; thumbs-down feedback
down-weights an episode so bad answers are not recalled as precedent.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from app.config import Settings
from app.db import Database
from app.errors import AuthorizationError
from app.llm.service import LLMService
from app.logging_config import get_logger
from app.retrieval.sparse import tokenize
from app.security.rbac import UserContext

log = get_logger(__name__)
_PREF = re.compile(r"\b(remember (that|this)|my preference is|i prefer|always (answer|respond|reply|use)|from now on)\b(.{3,200})", re.I | re.S)
_LEVEL_RANK = {"public": 0, "internal": 1, "confidential": 2, "restricted": 3}


@dataclass
class Turn:
    role: str
    content: str
    ts: float = field(default_factory=time.time)


@dataclass
class Session:
    session_id: str
    user_id: str
    turns: list[Turn] = field(default_factory=list)
    summary: str = ""
    created: float = field(default_factory=time.time)


class MemoryManager:
    def __init__(self, settings: Settings, db: Database, llm: LLMService):
        self.s, self.db, self.llm = settings, db, llm
        self._sessions: dict[str, Session] = {}

    # ------------------------------------------------------------------ sessions
    def get_session(self, session_id: str, user: UserContext) -> Session:
        sess = self._sessions.get(session_id)
        if sess is None:
            if len(self._sessions) > 500:  # crude eviction of the oldest sessions
                for sid in sorted(self._sessions, key=lambda k: self._sessions[k].created)[:100]:
                    self._sessions.pop(sid, None)
            sess = self._sessions[session_id] = Session(session_id, user.user_id)
        elif sess.user_id != user.user_id:
            raise AuthorizationError("This session belongs to another user.")
        return sess

    async def build_context(self, user: UserContext, session: Session, question: str) -> dict:
        recent = session.turns[-self.s.memory_recent_turns * 2:]
        long_term = await self.recall(user, question, self.s.memory_recall_k)
        prefs = await self.preferences(user)
        return {
            "recent_turns": [{"role": t.role, "content": t.content[:1200]} for t in recent],
            "summary": session.summary,
            "long_term": long_term,
            "preferences": prefs,
            "profile": {"name": user.name, "role": user.role.value, "department": user.department},
            "turn_count": len(session.turns) // 2,
        }

    async def record_turn(self, session: Session, question: str, answer: str) -> bool:
        session.turns.append(Turn("user", question))
        session.turns.append(Turn("assistant", answer))
        return await self._compress_if_needed(session)

    async def _compress_if_needed(self, session: Session) -> bool:
        limit = self.s.memory_compress_after_turns * 2
        if len(session.turns) <= limit:
            return False
        keep = self.s.memory_recent_turns * 2
        old, session.turns = session.turns[:-keep], session.turns[-keep:]
        try:
            session.summary = (await self.llm.complete(
                "Summarise this conversation so far in <=120 words. Keep entities, document ids, decisions and open questions.",
                f"Previous summary: {session.summary}\n\n" + "\n".join(f"{t.role}: {t.content[:500]}" for t in old),
                task="summarize_history", tier="fast",
                ctx={"previous_summary": session.summary, "turns": [{"content": t.content} for t in old]})).strip()
        except Exception as exc:  # noqa: BLE001 - degrade: keep a crude summary
            log.warning("memory_compress_failed", error=str(exc)[:120])
            session.summary = (session.summary + " " + " ".join(t.content[:80] for t in old))[-900:]
        return True

    # ------------------------------------------------------------------ long-term memory
    @staticmethod
    def extract_preferences(question: str) -> list[str]:
        return [m.group(0).strip()[:240] for m in _PREF.finditer(question)]

    async def remember(self, user: UserContext, kind: str, content: str, *, access_level: str = "internal",
                       session_id: str | None = None, message_id: str | None = None) -> None:
        kw = " ".join(sorted(set(tokenize(content))))
        await self.db.execute(
            "INSERT INTO memories(user_id,kind,content,keywords,access_level,session_id,message_id,weight,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (user.user_id, kind, content[:600], kw, access_level, session_id, message_id, 1.0, time.time()))

    async def recall(self, user: UserContext, query: str, k: int) -> list[dict]:
        rows = await self.db.fetchall("SELECT * FROM memories WHERE user_id=? AND kind='episode' ORDER BY id DESC LIMIT 200", (user.user_id,))
        q = set(tokenize(query))
        if not q:
            return []
        allowed = set(user.allowed_access_levels)
        scored = []
        for r in rows:
            if r["access_level"] not in allowed:  # role may have been downgraded since the episode was stored
                continue
            kw = set(r["keywords"].split())
            overlap = len(q & kw) / max(1, len(q | kw))
            age_days = (time.time() - r["created_at"]) / 86400
            score = overlap * (0.5 ** (age_days / 30)) * r["weight"]
            if score > 0.04:
                scored.append((score, r))
        scored.sort(key=lambda x: -x[0])
        return [{"content": r["content"], "score": round(s, 3), "message_id": r["message_id"]} for s, r in scored[:k]]

    async def preferences(self, user: UserContext) -> list[str]:
        rows = await self.db.fetchall("SELECT content FROM memories WHERE user_id=? AND kind='preference' ORDER BY id DESC LIMIT 5", (user.user_id,))
        return [r["content"] for r in rows]

    async def apply_feedback(self, user: UserContext, message_id: str, rating: int) -> None:
        factor = 1.25 if rating > 0 else 0.3 if rating < 0 else 1.0
        await self.db.execute("UPDATE memories SET weight = MIN(weight * ?, 3.0) WHERE user_id=? AND message_id=?", (factor, user.user_id, message_id))

    async def snapshot(self, user: UserContext, session: Session) -> dict:
        longterm = await self.db.fetchall("SELECT kind, content, access_level, weight FROM memories WHERE user_id=? ORDER BY id DESC LIMIT 15", (user.user_id,))
        return {"turns": len(session.turns) // 2, "summary": session.summary, "recent": [t.__dict__ for t in session.turns[-6:]],
                "long_term": [m for m in longterm if m["access_level"] in user.allowed_access_levels]}


def max_access_level(levels: list[str]) -> str:
    return max(levels, key=lambda l: _LEVEL_RANK.get(l, 1)) if levels else "internal"

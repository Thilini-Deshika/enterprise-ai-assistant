"""Offline "LLM": deterministic task-specific heuristics so the full pipeline runs without API keys.

It is NOT trying to be smart - it exercises every code path (routing, RLM map/reduce, citations,
validation, streaming) so the system can be developed, unit-tested and demoed offline, and it is
the proof that the architecture does not depend on one model. Real providers replace it 1:1.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from datetime import date
from typing import AsyncIterator

from app.agents import heuristics as H
from app.config import MOCK_EMBEDDING_DIM


def hash_embed(text: str, dim: int = MOCK_EMBEDDING_DIM) -> list[float]:
    vec = [0.0] * dim
    for tok in re.findall(r"[a-z0-9]+", text.lower()):
        h = int(hashlib.md5(tok.encode()).hexdigest()[:8], 16)
        vec[h % dim] += 1.0 if (h >> 31) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class MockBackend:
    async def complete(self, task: str, system: str, user: str, ctx: dict) -> str:
        fn = getattr(self, f"_task_{task}", None)
        if fn is None:
            return json.dumps({}) if task.endswith("_json") else "OK"
        return fn(system, user, ctx)

    async def stream(self, task: str, system: str, user: str, ctx: dict) -> AsyncIterator[str]:
        text = await self.complete(task, system, user, ctx)
        for part in re.findall(r"\S+\s*", text):
            await asyncio.sleep(0.004)
            yield part

    # ---- supervisor / routing -------------------------------------------------------------
    def _task_supervisor(self, system, user, ctx):
        q = ctx["question"]
        intent = H.classify_intent(q)
        filters = H.extract_filters(q, date.fromisoformat(ctx["today"]))
        return json.dumps({"intent": intent, "standalone_query": ctx.get("standalone_query") or q,
                           "filters": filters, "plan": H.default_plan(intent),
                           "reasoning": f"[offline heuristic] classified as {intent}"})

    def _task_rewrite(self, system, user, ctx):
        q, history = ctx["question"], ctx.get("history", [])
        if history and re.search(r"\b(it|that|this|those|them|previous|same|again|more)\b", q, re.I) and len(q.split()) < 12:
            return f"{history[-1][:160]} {q}"
        return q

    def _task_summarize_history(self, system, user, ctx):
        return (ctx.get("previous_summary", "") + " " + " | ".join(t["content"][:120] for t in ctx.get("turns", [])))[-900:]

    # ---- tools ------------------------------------------------------------------------------
    def _task_tool_select(self, system, user, ctx):
        return json.dumps(H.pick_tool_calls(ctx["question"], ctx["intent"], set(ctx["available"]), ctx["prior"], ctx["filters"]))

    # ---- RLM -----------------------------------------------------------------------------------
    def _task_rlm_plan(self, system, user, ctx):
        from app.agents.rlm.planner import default_plan_code
        return default_plan_code(ctx["question"], ctx["filters"])

    def _task_rlm_map(self, system, user, ctx):
        lines = []
        for block in ctx["context"].split("=== DOC ")[1:]:
            head = block.split("\n", 1)[0]
            doc_id = head.split(" | ")[0].strip()
            when = re.search(r"date=(\S+)", head)
            cause_line = None
            for m in re.finditer(r"\[(\S+#\d+)\] \(([^)]*)\) (.+)", block):
                cid, section, text = m.groups()
                if re.search(r"root cause", section, re.I):
                    cause_line = (cid, re.split(r"(?<=[.!?])\s", text.strip())[0])
                    break
            if cause_line is None:
                m = re.search(r"\[(\S+#\d+)\] \(([^)]*)\) (.+)", block)
                if m:
                    cause_line = (m.group(1), re.split(r"(?<=[.!?])\s", m.group(3).strip())[0])
            if cause_line:
                lines.append(f"- {doc_id} ({when.group(1) if when else 'n/a'}): {cause_line[1]} [{cause_line[0]}]")
        return "\n".join(lines) or "- no relevant findings"

    def _task_rlm_reduce(self, system, user, ctx):
        groups: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for line in ctx["context"].splitlines():
            m = re.match(r"- (\S+) \(([^)]*)\): (.+?) \[(\S+#\d+)\]\s*$", line.strip())
            if m:
                doc_id, _, cause, cid = m.groups()
                groups[_cause_key(cause)].append((doc_id, cid, cause))
        if not groups:
            return ctx["context"][:1500]
        total = sum(len(v) for v in groups.values())
        out = [f"{total} incidents analysed; recurring root causes (most frequent first):"]
        for i, (key, items) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
            refs = ", ".join(f"{d} [{c}]" for d, c, _ in items[:6])
            out.append(f"{i}. {items[0][2].rstrip('.')} - {len(items)} incident(s): {refs}")
        return "\n".join(out)

    # ---- answer ----------------------------------------------------------------------------------
    def _task_answer(self, system, user, ctx):
        q = ctx["question"]
        if ctx.get("notes"):
            notes = " ".join(ctx["notes"])
        else:
            notes = ""
        if (ctx.get("research") or {}).get("summary"):
            return f"{notes}Here is what the investigation across the document set found.\n\n{ctx['research']['summary']}".strip()
        parts = []
        for r in ctx.get("tool_results", []):
            if r["status"] == "ok":
                parts.append(f"The `{r['tool']}` tool returned: {json.dumps(r['data'], default=str)[:600]} [{r['id']}]")
            else:
                parts.append(f"`{r['tool']}` could not be used ({r['status']}): {r.get('error', '')[:160]}")
        ev = ctx.get("evidence", [])[:3]
        for e in ev:
            first = re.split(r"(?<=[.!?])\s", e["text"].strip())
            snippet = " ".join(first[:2])[:380]
            parts.append(f"{e['title']} ({e['section']}): {snippet} [{e['chunk_id']}]")
        if not parts:
            if ctx.get("intent") == "chitchat":
                from app.config import get_settings
                s = get_settings()
                return (f"Hello, I am {s.assistant_name}, the internal knowledge assistant of {s.org_name}. "
                        "I can search policies, runbooks, incident reports and architecture documents, and look up service and employee data.")
            return (f"{notes}I could not find supporting material for \"{q[:120]}\" in the documents available to you. "
                    "Try rephrasing, naming a service or document type, or widening the date range.")
        return notes + "\n\n".join(parts)

    def _task_judge(self, system, user, ctx):
        return json.dumps({"faithful": True, "unsupported_claims": []})


def _cause_key(cause: str) -> str:
    c = cause.lower()
    for key in ("connection pool", "tls certificate", "certificate", "processor", "fraud rule", "configuration", "consumer lag", "queue"):
        if key in c:
            return key
    return re.sub(r"[^a-z ]", "", c)[:40]

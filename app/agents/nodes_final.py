"""Back-of-graph agents: response generation, validation (guardrails), memory write."""
from __future__ import annotations

import json
import re

from app.agents.common import instrument, user_of
from app.agents.deps import AgentDeps
from app.brand import system_prompt
from app.errors import LLMUnavailable
from app.events import emit
from app.llm.mock_backend import MockBackend
from app.logging_config import get_logger
from app.memory.manager import max_access_level
from app.security import injection
from app.security.guardrails import CITATION_RE, CheckResult, CitationVerifier, extract_citations

log = get_logger("agents.final")
MAX_ATTEMPTS = 2


# ------------------------------------------------------------------------------------ response
def select_evidence(state: dict, max_chunks: int = 8, max_chars: int = 7000) -> list[dict]:
    research = state.get("research") or {}
    cited = set(extract_citations(research.get("summary", "")))
    if cited:
        max_chunks = 14
    ordered = sorted(state.get("evidence", []), key=lambda e: (e["chunk_id"] not in cited, -e.get("rerank", 0)))
    out, used = [], 0
    for e in ordered:
        if len(out) >= max_chunks or used + len(e["text"]) > max_chars:
            break
        out.append(e)
        used += len(e["text"])
    return out


def build_user_prompt(state: dict, evidence: list[dict], feedback: str) -> str:
    mem, parts = state.get("memory", {}), []
    prof = mem.get("profile", {})
    parts.append(injection.spotlight("memory", json.dumps({
        "user": prof, "earlier_conversation_summary": mem.get("summary", ""), "preferences": mem.get("preferences", []),
        "possibly_relevant_past_questions": [m["content"][:240] for m in mem.get("long_term", [])]}, ensure_ascii=False)))
    if state.get("notes"):
        parts.append("SYSTEM NOTES (trusted, mention limitations to the user where relevant):\n- " + "\n- ".join(dict.fromkeys(state["notes"])))
    if evidence:
        parts.append("EVIDENCE:\n" + "\n".join(injection.spotlight("document", e["text"], id=e["chunk_id"], title=e["title"], section=e["section"],
                                                                  date=e["metadata"].get("created_date", ""), access=e["metadata"].get("access_level", ""))
                                                 for e in evidence))
    if state.get("research"):
        r = state["research"]
        parts.append(injection.spotlight("research", r["summary"], documents_analysed=r["stats"].get("documents_analysed"), status=r["status"]))
    for r in state.get("tool_results", []):
        if r["tool"] == "knowledge_search":
            continue
        body = json.dumps(r["data"], default=str)[:2500] if r["ok"] else f"UNAVAILABLE ({r['status']}): {r['error']}"
        parts.append(injection.spotlight("tool_result", body, id=r["id"], tool=r["tool"], status=r["status"]))
    prompt = "\n\n".join(parts) + f"\n\nQUESTION: {state['question']}"
    if feedback:
        prompt += f"\n\nYOUR PREVIOUS DRAFT FAILED VALIDATION. Fix these problems and answer again:\n{feedback}"
    return prompt


def make_response(deps: AgentDeps):
    @instrument("response", on_fail=lambda st: {"attempts": st.get("attempts", 0) + 1, "draft": ""})
    async def response(state: dict) -> dict:
        attempts = state.get("attempts", 0)
        feedback = (state.get("validation") or {}).get("feedback", "") if attempts else ""
        evidence = select_evidence(state)
        system = system_prompt(deps.settings.canary_token, deps.settings.org_name, deps.settings.assistant_name)
        user_prompt = build_user_prompt(state, evidence, feedback)
        offline = {"question": state["question"], "intent": state.get("intent"), "evidence": evidence, "research": state.get("research"),
                   "tool_results": [r for r in state.get("tool_results", []) if r["tool"] != "knowledge_search"], "notes": []}
        if attempts:
            emit("token_reset", reason="regenerating after validation feedback")
        emit("agent_state", agent="response", status="streaming", detail=f"attempt {attempts + 1}: {len(evidence)} evidence chunks in context")
        draft, status = "", "ok"
        try:
            async for tok in deps.llm.stream(system, user_prompt, task="answer", tier="strong", ctx=offline):
                draft += tok
                emit("token", text=tok)
        except LLMUnavailable as exc:
            status = "degraded"
            emit("agent_state", agent="response", status="degraded", detail="LLM unavailable; composing an extractive answer from retrieved evidence")
            emit("token_reset", reason="LLM unavailable")
            offline["notes"] = ["(Automated summary - the language model was unavailable.) "]
            draft = await MockBackend().complete("answer", "", "", offline)
            emit("token", text=draft)
        return {"draft": draft, "attempts": attempts + 1, "agent_status": {"response": status}}
    return response


# ------------------------------------------------------------------------------------ validator
JUDGE_SYSTEM = ("You are a strict fact-checker. Given EVIDENCE/TOOL RESULTS and an ANSWER, return ONLY JSON "
                '{"faithful": true|false, "unsupported_claims": ["..."]}. faithful=true only if every factual claim in the ANSWER is supported by the evidence.')


def source_lines(cited: list[str], evidence: dict[str, dict], tools: dict[str, dict]) -> tuple[list[dict], str]:
    cites, lines = [], []
    for cid in cited:
        if cid in evidence:
            e = evidence[cid]
            md = e["metadata"]
            cites.append({"id": cid, "type": "document", "doc_id": e["doc_id"], "title": e["title"], "section": e["section"],
                          "date": md.get("created_date"), "access_level": md.get("access_level"), "snippet": e["text"][:220]})
            lines.append(f"- [{cid}] {e['title']} - {e['section']} ({md.get('created_date')}, {md.get('access_level')})")
        elif cid in tools:
            cites.append({"id": cid, "type": "tool", "title": tools[cid]["tool"], "section": tools[cid]["status"]})
            lines.append(f"- [{cid}] result of tool `{tools[cid]['tool']}`")
    return cites, ("\n\n**Sources**\n" + "\n".join(lines)) if lines else ""


SAFE_FAILURE_ANSWER = ("I'm sorry - I couldn't complete a verified answer to that right now. No unverified information has been shown. "
                       "Please try again in a moment or rephrase the question.")


def _validator_failed(state: dict) -> dict:
    """If the validator itself crashes we fail CLOSED: show a safe message, never the unvalidated draft."""
    emit("final", message_id=state["message_id"], answer=SAFE_FAILURE_ANSWER, citations=[], degraded=True, blocked=False,
         validation={"passed": True, "checks": [], "degraded": True})
    return {"answer": SAFE_FAILURE_ANSWER, "citations": [], "degraded": True,
            "validation": {"passed": True, "checks": [], "degraded": True}}


def make_validator(deps: AgentDeps):
    @instrument("validator", on_fail=_validator_failed)
    async def validator(state: dict) -> dict:
        user, draft, attempts = user_of(state), state.get("draft", ""), state.get("attempts", 1)
        evidence = {e["chunk_id"]: e for e in state.get("evidence", [])}
        tools = {r["id"]: r for r in state.get("tool_results", []) if r["ok"] and r["tool"] != "knowledge_search"}
        required = bool(evidence or tools) and state.get("intent") != "chitchat"
        all_checks: list[CheckResult] = []

        # 1) citations: exist, authorised, present, grounded
        cit_checks, invalid = CitationVerifier(evidence, set(tools), user.allowed_access_levels).verify(draft, citations_required=required)
        emit("validation", stage="citations", passed=all(c.passed for c in cit_checks if c.severity == "error"), checks=[c.to_dict() for c in cit_checks])
        # 2) safety: canary / secrets / PII redaction
        out_checks, cleaned = deps.output_guard.check(draft)
        emit("validation", stage="safety", passed=all(c.passed for c in out_checks), checks=[c.to_dict() for c in out_checks])
        # 3) brand + response validity
        brand_checks, cleaned = deps.brand_guard.check(cleaned)
        emit("validation", stage="brand", passed=all(c.passed for c in brand_checks), checks=[c.to_dict() for c in brand_checks])
        all_checks += cit_checks + out_checks + brand_checks
        # 4) LLM faithfulness judge (skipped offline / when disabled / nothing to check against)
        if deps.settings.llm_judge_enabled and deps.llm.provider != "mock" and (evidence or tools) and not invalid:
            try:
                ev_text = "\n".join(f"[{e['chunk_id']}] {e['text'][:600]}" for e in list(evidence.values())[:12])
                verdict = await deps.llm.json(JUDGE_SYSTEM, f"EVIDENCE:\n{ev_text}\n\nTOOL RESULTS: {json.dumps([t['data'] for t in tools.values()], default=str)[:1500]}\n\nANSWER:\n{cleaned}",
                                              task="judge", tier="fast", ctx={})
                ok = bool(verdict.get("faithful", True))
                all_checks.append(CheckResult("faithfulness_judge", ok, "all claims supported" if ok else f"unsupported: {verdict.get('unsupported_claims', [])[:2]}"))
            except Exception as exc:  # noqa: BLE001
                all_checks.append(CheckResult("faithfulness_judge", True, f"judge unavailable ({type(exc).__name__}); skipped", severity="warning"))
            emit("validation", stage="faithfulness", passed=all_checks[-1].passed, checks=[all_checks[-1].to_dict()])

        failed = [c for c in all_checks if not c.passed and c.severity == "error"]
        degraded = bool(state.get("degraded")) or state.get("agent_status", {}).get("response") == "degraded"
        if failed and attempts < MAX_ATTEMPTS:
            feedback = "; ".join(f"{c.name}: {c.detail}" for c in failed)
            emit("agent_state", agent="validator", status="retry", detail=f"validation failed ({feedback[:160]}); sending back to the response agent")
            return {"validation": {"passed": False, "feedback": feedback, "checks": [c.to_dict() for c in all_checks]}}
        if failed:  # attempts exhausted -> repair or fall back to a safe, fully-verified answer
            degraded = True
            for bad in invalid:
                cleaned = cleaned.replace(f"[{bad}]", "")
            still = [c.name for c in failed if c.name not in ("citations_exist",)]
            if still or not extract_citations(cleaned) and required:
                top = list(evidence.values())[:3]
                cleaned = ("I wasn't able to produce an answer I could fully verify against the documents available to you. "
                           + ("The most relevant sources I found are listed below; please review them directly." if top else
                              "Please try rephrasing the question or narrowing it by service, document type or date.")
                           + "".join(f" [{e['chunk_id']}]" for e in top))
            emit("guardrail", kind="answer_repaired", failed=[c.name for c in failed])
        cited = extract_citations(cleaned)
        citations, sources = source_lines(cited, evidence, tools)
        note = ""
        if state.get("notes") and degraded is False and any("degraded" in n or "unavailable" in n.lower() for n in state["notes"]):
            note = "\n\n_Note: " + " ".join(dict.fromkeys(n for n in state["notes"] if "degraded" in n or "unavailable" in n.lower()))[:400] + "_"
        final = cleaned.strip() + note + sources
        validation = {"passed": True, "checks": [c.to_dict() for c in all_checks], "degraded": degraded}
        emit("final", message_id=state["message_id"], answer=final, citations=citations, degraded=degraded, blocked=False,
             validation=validation, intent=state.get("intent"), agent_status=state.get("agent_status", {}),
             research_stats=(state.get("research") or {}).get("stats"))
        return {"answer": final, "citations": citations, "validation": validation, "degraded": degraded,
                "agent_status": {"validator": "degraded" if degraded else "ok"}}
    return validator


# ------------------------------------------------------------------------------------ memory write
def make_memory_write(deps: AgentDeps):
    @instrument("memory_write")
    async def memory_write(state: dict) -> dict:
        user = user_of(state)
        session = deps.memory.get_session(state["session_id"], user)
        plain = re.sub(r"\n\n\*\*Sources\*\*.*$", "", state["answer"], flags=re.S)
        compressed = await deps.memory.record_turn(session, state["question"], plain)
        written = []
        cites = state.get("citations", [])
        if state.get("validation", {}).get("passed") and not state.get("degraded") and cites:
            level = max_access_level([c.get("access_level") or "internal" for c in cites if c["type"] == "document"])
            await deps.memory.remember(user, "episode", f"Q: {state['question'][:200]} | A: {plain[:300]} | docs: {', '.join(c['id'] for c in cites[:5])}",
                                       access_level=level, session_id=state["session_id"], message_id=state["message_id"])
            written.append("episode")
        for pref in deps.memory.extract_preferences(state["question"]):
            await deps.memory.remember(user, "preference", pref, session_id=state["session_id"])
            written.append("preference")
        emit("memory", action="write", turns=len(session.turns) // 2, compressed_summary=compressed, stored=written,
             summary_chars=len(session.summary))
        return {}
    return memory_write

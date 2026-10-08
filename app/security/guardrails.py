"""Output-side guardrails. Run by the Validator agent on every draft answer.

  CitationVerifier  - no hallucinated citations, all cited evidence was authorised for this user,
                      cited text actually supports the sentence (lexical grounding check)
  OutputGuard       - canary/system-prompt leak, secrets, card-number (Luhn) redaction
  BrandGuard        - banned phrases, regulatory-advice disclaimer, length sanity
Each check returns a CheckResult so the UI can show a per-check validation panel.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.brand import BANNED_PHRASES, REGULATORY_DISCLAIMER, REGULATORY_HINTS

CITATION_RE = re.compile(r"\[((?:tool:[a-z_]+#\d+)|(?:[A-Za-z0-9][A-Za-z0-9_.\-]*#\d+))\]")
_STOP = frozenset("the a an and or of to in on for with by is are was were be been this that these those it its as at from "
                  "which who whom into than then also not no but if so such can could should would will may might has have had "
                  "their there they them our your you we i he she his her".split())
_SECRET_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bpc[a-z]{0,3}-[A-Za-z0-9\-]{30,}\b"),
    re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key)\s*[:=]\s*\S{6,}"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
]
_CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")


@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""
    severity: str = "error"  # error -> triggers regeneration; warning -> reported only

    def to_dict(self) -> dict:
        return {"name": self.name, "passed": self.passed, "detail": self.detail, "severity": self.severity}


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 3 and t not in _STOP}


def extract_citations(text: str) -> list[str]:
    return list(dict.fromkeys(CITATION_RE.findall(text)))


class CitationVerifier:
    def __init__(self, evidence: dict[str, dict], tool_ids: set[str], allowed_levels: tuple[str, ...]):
        self.evidence, self.tool_ids, self.allowed_levels = evidence, tool_ids, allowed_levels

    def verify(self, answer: str, *, citations_required: bool) -> tuple[list[CheckResult], list[str]]:
        cited = extract_citations(answer)
        valid_ids = set(self.evidence) | self.tool_ids
        invalid = [c for c in cited if c not in valid_ids]
        unauthorised = [c for c in cited if c in self.evidence and self.evidence[c]["metadata"].get("access_level") not in self.allowed_levels]
        results = [
            CheckResult("citations_exist", not invalid, f"hallucinated: {invalid}" if invalid else f"{len(cited)} citation(s) verified against retrieved evidence"),
            CheckResult("citations_authorised", not unauthorised, f"not permitted: {unauthorised}" if unauthorised else "all cited documents are within the user's access level"),
        ]
        if citations_required:
            results.append(CheckResult("citations_present", bool(cited), "answer uses evidence but cites none" if not cited else "at least one citation present"))
        results.append(self._grounding(answer))
        return results, invalid

    def _grounding(self, answer: str) -> CheckResult:
        """Each sentence carrying a citation should share vocabulary with the cited chunk."""
        weak = []
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
            ids = [c for c in CITATION_RE.findall(sentence) if c in self.evidence]
            claim = _tokens(CITATION_RE.sub("", sentence))
            if not ids or len(claim) < 4:
                continue
            support = max(len(claim & _tokens(self.evidence[i]["text"] + " " + self.evidence[i]["title"])) / len(claim) for i in ids)
            if support < 0.2:
                weak.append(sentence.strip()[:80])
        return CheckResult("citation_grounding", not weak, f"weakly supported: {weak[:2]}" if weak else "cited text supports the claims (lexical check)", severity="warning")


class OutputGuard:
    def __init__(self, canary: str):
        self.canary = canary

    def check(self, answer: str) -> tuple[list[CheckResult], str]:
        results, cleaned = [], answer
        leaked = self.canary in answer or "NON-NEGOTIABLE RULES" in answer
        results.append(CheckResult("no_prompt_leak", not leaked, "system prompt / canary leaked" if leaked else "no system prompt leakage"))
        secrets = [p.pattern[:20] for p in _SECRET_PATTERNS if p.search(answer)]
        results.append(CheckResult("no_secrets", not secrets, "credential-like string in output" if secrets else "no credential patterns"))
        redacted = 0

        def _mask(m: re.Match) -> str:
            nonlocal redacted
            digits = re.sub(r"\D", "", m.group(0))
            if 13 <= len(digits) <= 19 and _luhn(digits):
                redacted += 1
                return "[REDACTED CARD NUMBER]"
            return m.group(0)

        cleaned = _CARD_RE.sub(_mask, cleaned)
        results.append(CheckResult("pii_redaction", True, f"{redacted} card number(s) redacted" if redacted else "no card numbers found", severity="warning"))
        return results, cleaned


class BrandGuard:
    def check(self, answer: str) -> tuple[list[CheckResult], str]:
        lowered = answer.lower()
        banned = [p for p in BANNED_PHRASES if p in lowered]
        results = [
            CheckResult("brand_voice", not banned, f"off-brand phrase(s): {banned}" if banned else "tone and wording on-brand"),
            CheckResult("response_valid", 20 <= len(answer) <= 12000, f"length={len(answer)}"),
        ]
        out = answer
        if any(h in lowered for h in REGULATORY_HINTS) and "not legal or regulatory advice" not in lowered:
            out += REGULATORY_DISCLAIMER
            results.append(CheckResult("regulatory_disclaimer", True, "disclaimer appended", severity="warning"))
        return results, out


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total, alt = total + d, not alt
    return total % 10 == 0

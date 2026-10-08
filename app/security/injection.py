"""Prompt-injection detection and neutralisation.

Layered defence:
  L1  this module   - pattern + heuristic scoring on user input, retrieved chunks and tool output
  L2  spotlighting  - untrusted text is wrapped in <document>/<tool_result> tags and the system prompt
                      states that tag contents are DATA (response agent)
  L3  capability    - even a fully hijacked LLM cannot exceed its role: tools are RBAC-checked,
                      write tools need human approval, retrieval filters are built server-side
  L4  output guards - citation / canary / secret / PII checks before anything reaches the user
Pattern matching is intentionally conservative and explainable; it is a tripwire, not the only wall.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_ZERO_WIDTH = dict.fromkeys(map(ord, "\u200b\u200c\u200d\u2060\ufeff"), None)

# category -> (weight, patterns)
_RULES: dict[str, tuple[float, list[str]]] = {
    "instruction_override": (0.9, [
        r"\b(ignore|disregard|forget|override|bypass)\b.{0,25}\b(all |any |the |your |these )?(previous|prior|above|earlier|system|safety|original)\b.{0,25}\b(instructions?|rules?|prompts?|guidelines?|polic(y|ies)|restrictions?)\b",
        r"\bnew (system )?instructions?\s*:",
        r"\byou are now\b(?! (going|able|ready))",
        r"\b(act|behave|respond) as (if you (are|were)|an?) (unrestricted|unfiltered|jailbroken|dan)\b",
        r"\b(developer|god|debug|sudo) mode\b",
        r"\bjailbreak\b",
        r"\bdo anything now\b",
    ]),
    "prompt_exfiltration": (0.85, [
        r"\b(reveal|show|print|repeat|output|display|leak|tell me)\b.{0,30}\b(your|the)\b.{0,15}\b(system|hidden|initial|original|secret)\b.{0,15}\b(prompt|instructions?|message|rules)\b",
        r"\bwhat (are|were) your (initial|system|hidden) (instructions?|prompt)\b",
        r"\brepeat everything above\b",
    ]),
    "data_exfiltration": (0.8, [
        r"\b(send|post|email|upload|forward|exfiltrate|transmit)\b.{0,80}(https?://|\bwebhook\b|@[a-z0-9-]+\.[a-z]{2,})",
        r"!\[[^\]]*\]\(https?://[^)]*\)",
        r"\b(dump|export|list|print|reveal|show me)\b.{0,15}\b(all )?(the )?(api[ _-]?keys?|passwords?|secrets?|credentials?|tokens?|customer (data|records|list)|connection strings?)\b",
    ]),
    "tool_abuse": (0.9, [
        r"\b(bypass|skip|disable|circumvent)\b.{0,20}\b(approval|authori[sz]ation|permissions?|rbac|guardrails?|validation|safety)\b",
        r"\b(delete|drop|truncate|wipe)\b.{0,20}\b(tables?|database|index|namespace|logs?|audit)\b",
        r"\b(os\.system|subprocess|__import__|eval\s*\(|exec\s*\(|rm\s+-rf|curl\s+http)",
        r"\bcall\b.{0,30}\b(tool|function)\b.{0,40}\bwithout\b.{0,15}\b(approval|permission|asking)\b",
    ]),
    "role_escalation": (0.6, [
        r"\b(i am|i'm|treat me as|pretend i am|assume i am)\b.{0,15}\b(the )?(admin|administrator|root|superuser|ceo|cto)\b",
        r"\bgrant me\b.{0,25}\b(access|admin|permissions?)\b",
    ]),
}
_COMPILED = {k: (w, [re.compile(p, re.I | re.S) for p in pats]) for k, (w, pats) in _RULES.items()}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?\n])\s+")


@dataclass
class InjectionReport:
    score: float = 0.0
    categories: list[str] = field(default_factory=list)
    matches: list[str] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return self.score > 0

    def to_dict(self) -> dict:
        return {"score": round(self.score, 2), "categories": self.categories, "matches": self.matches[:5]}


def normalize(text: str) -> str:
    """Defeat trivial obfuscation: unicode look-alikes, zero-width chars, odd spacing."""
    text = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    return re.sub(r"\s+", " ", text)


def scan(text: str) -> InjectionReport:
    norm = normalize(text)
    report = InjectionReport()
    hits = 0
    for category, (weight, patterns) in _COMPILED.items():
        for pat in patterns:
            m = pat.search(norm)
            if m:
                hits += 1
                if category not in report.categories:
                    report.categories.append(category)
                report.matches.append(m.group(0)[:80])
                report.score = max(report.score, weight)
    if hits > 1:  # several independent signals -> more suspicious
        report.score = min(1.0, report.score + 0.05 * (hits - 1))
    return report


def sanitize_untrusted(text: str, threshold: float = 0.5) -> tuple[str, list[str]]:
    """Remove only the offending sentences from retrieved/tool text (keeps the legitimate content)."""
    kept, removed = [], []
    for sentence in _SENTENCE_SPLIT.split(text):
        if sentence and scan(sentence).score >= threshold:
            removed.append(sentence.strip()[:160])
            kept.append("[removed: suspected instruction injection]")
        else:
            kept.append(sentence)
    return " ".join(kept), removed


def spotlight(tag: str, text: str, **attrs) -> str:
    """Wrap untrusted content in explicit delimiters (a.k.a. spotlighting / datamarking)."""
    safe = text.replace("</", "<\u200b/")  # stop content from closing our tag
    attr_str = "".join(f' {k}="{str(v).replace(chr(34), chr(39))}"' for k, v in attrs.items())
    return f"<{tag}{attr_str}>\n{safe}\n</{tag}>"

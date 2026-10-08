"""Provider-agnostic LLM + embedding service.

* OpenAI / Azure OpenAI through langchain-openai, so every call becomes a child span in LangSmith.
* `mock` provider (no keys) for offline development, tests and CI.
* Resilience: per-call timeout, jittered retries, circuit breaker, and tier fallback
  (strong -> fast) before surfacing LLMUnavailable. Callers degrade to heuristics on failure.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import AsyncIterator

from app.config import Settings
from app.errors import DependencyUnavailable, LLMUnavailable
from app.llm.mock_backend import MockBackend, hash_embed
from app.logging_config import get_logger
from app.resilience import CircuitBreaker, retry_async

log = get_logger(__name__)


def parse_json_loose(text: str) -> dict:
    """Parse an LLM JSON reply that may be wrapped in ```fences``` or surrounded by prose."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


class LLMService:
    def __init__(self, settings: Settings):
        self.s = settings
        self.provider = settings.resolved_llm_provider
        self.breaker = CircuitBreaker("llm", failure_threshold=4, reset_after_s=20)
        self.embed_breaker = CircuitBreaker("embeddings", failure_threshold=4, reset_after_s=20)
        self._mock = MockBackend() if self.provider == "mock" else None
        self._chat_cache: dict[tuple, object] = {}
        self._embedder = None
        self.dim = settings.effective_embedding_dim

    # ------------------------------------------------------------------ model construction
    def _model_name(self, tier: str) -> str:
        if self.provider == "azure":
            return self.s.azure_deployment_strong if tier == "strong" else self.s.azure_deployment_fast
        return self.s.llm_model_strong if tier == "strong" else self.s.llm_model_fast

    def _chat(self, tier: str, json_mode: bool, temperature: float):
        key = (tier, json_mode, temperature)
        if key not in self._chat_cache:
            kwargs = dict(timeout=self.s.llm_timeout_s, max_retries=0, temperature=temperature)
            if self.provider == "azure":
                from langchain_openai import AzureChatOpenAI
                model = AzureChatOpenAI(azure_deployment=self._model_name(tier), api_version=self.s.azure_openai_api_version,
                                        azure_endpoint=self.s.azure_openai_endpoint, api_key=self.s.azure_openai_api_key, **kwargs)
            else:
                from langchain_openai import ChatOpenAI
                model = ChatOpenAI(model=self._model_name(tier), api_key=self.s.openai_api_key,
                                   base_url=self.s.openai_base_url or None, **kwargs)
            if json_mode:
                model = model.bind(response_format={"type": "json_object"})
            self._chat_cache[key] = model
        return self._chat_cache[key]

    @staticmethod
    def _config(task: str, tier: str) -> dict:
        return {"run_name": f"llm:{task}", "tags": ["llm", f"task:{task}", f"tier:{tier}"], "metadata": {"task": task, "tier": tier}}

    # ------------------------------------------------------------------ completion
    async def complete(self, system: str, user: str, *, task: str, tier: str = "fast", json_mode: bool = False,
                       ctx: dict | None = None, temperature: float = 0.0) -> str:
        if self._mock:
            return await self._mock.complete(task, system, user, ctx or {})
        tiers = [tier] + (["fast"] if tier == "strong" else [])
        last: Exception | None = None
        for t in tiers:
            async def _call(t=t):
                msg = await asyncio.wait_for(
                    self._chat(t, json_mode, temperature).ainvoke([("system", system), ("user", user)], config=self._config(task, t)),
                    timeout=self.s.llm_timeout_s + 5)
                return msg.content if isinstance(msg.content, str) else "".join(str(p) for p in msg.content)
            try:
                return await self.breaker.call(lambda: retry_async(_call, attempts=self.s.llm_retries + 1, op=f"llm:{task}"))
            except Exception as exc:  # noqa: BLE001
                last = exc
                log.warning("llm_call_failed", task=task, tier=t, error=str(exc)[:200])
        raise LLMUnavailable(f"LLM unavailable for task '{task}': {last}") from last

    async def json(self, system: str, user: str, *, task: str, tier: str = "fast", ctx: dict | None = None) -> dict:
        raw = await self.complete(system, user, task=task, tier=tier, json_mode=True, ctx=ctx)
        try:
            return parse_json_loose(raw)
        except json.JSONDecodeError:
            repaired = await self.complete(system + "\nReturn ONLY one valid JSON object, nothing else.", user,
                                           task=f"{task}_repair", tier=tier, json_mode=True, ctx=ctx)
            return parse_json_loose(repaired)

    async def stream(self, system: str, user: str, *, task: str, tier: str = "strong", ctx: dict | None = None) -> AsyncIterator[str]:
        if self._mock:
            async for tok in self._mock.stream(task, system, user, ctx or {}):
                yield tok
            return
        model = self._chat(tier, False, 0.2)
        try:
            async for chunk in model.astream([("system", system), ("user", user)], config=self._config(task, tier)):
                piece = chunk.content if isinstance(chunk.content, str) else ""
                if piece:
                    yield piece
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(f"LLM streaming failed: {exc}") from exc

    # ------------------------------------------------------------------ embeddings
    def _get_embedder(self):
        if self._embedder is None:
            if self.provider == "azure":
                from langchain_openai import AzureOpenAIEmbeddings
                self._embedder = AzureOpenAIEmbeddings(azure_deployment=self.s.azure_deployment_embedding,
                                                       api_version=self.s.azure_openai_api_version,
                                                       azure_endpoint=self.s.azure_openai_endpoint, api_key=self.s.azure_openai_api_key)
            else:
                from langchain_openai import OpenAIEmbeddings
                self._embedder = OpenAIEmbeddings(model=self.s.embedding_model, api_key=self.s.openai_api_key,
                                                  base_url=self.s.openai_base_url or None, dimensions=self.s.embedding_dim
                                                  if self.s.embedding_model.startswith("text-embedding-3") else None)
        return self._embedder

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if self._mock:
            return [hash_embed(t) for t in texts]

        async def _call():
            return await asyncio.wait_for(self._get_embedder().aembed_documents(texts), timeout=self.s.llm_timeout_s)
        try:
            return await self.embed_breaker.call(lambda: retry_async(_call, attempts=self.s.llm_retries + 1, op="embed"))
        except Exception as exc:  # noqa: BLE001
            raise DependencyUnavailable(f"Embedding service unavailable: {exc}", dependency="embeddings") from exc

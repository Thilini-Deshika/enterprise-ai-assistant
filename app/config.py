"""Central configuration. Everything tunable lives here and can be overridden via env / .env."""
from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

# Export .env into os.environ so SDKs that read env directly (LangSmith, OpenAI) see the values.
load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
MOCK_EMBEDDING_DIM = 256


class RateLimitConfig(BaseModel):
    capacity: float
    refill_per_sec: float


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Enterprise AI Assistant"
    org_name: str = "Example Bank"             # organisation the assistant speaks for (persona)
    assistant_name: str = "Enterprise AI Assistant"
    log_level: str = "INFO"
    jwt_secret: str = "change-me-to-a-long-random-string-of-32+chars"
    jwt_ttl_minutes: int = 120
    canary_token: str = "CBA-CANARY-9c1f27"  # planted in the system prompt; must never appear in output

    # ---- LLM ----
    llm_provider: Literal["auto", "openai", "azure", "mock"] = "auto"
    openai_api_key: str = ""
    openai_base_url: str = ""
    llm_model_fast: str = "gpt-4o-mini"
    llm_model_strong: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    azure_openai_api_version: str = "2024-10-21"
    azure_deployment_fast: str = "gpt-4o-mini"
    azure_deployment_strong: str = "gpt-4o"
    azure_deployment_embedding: str = "text-embedding-3-small"
    llm_timeout_s: float = 45.0
    llm_retries: int = 2
    llm_judge_enabled: bool = True  # LLM faithfulness check in the validator (skipped in mock mode)

    # ---- Vector DB / retrieval ----
    vector_backend: Literal["auto", "pinecone", "local"] = "auto"
    pinecone_api_key: str = ""
    pinecone_index: str = "enterprise-kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    hybrid_alpha: float = 0.6
    hybrid_mode: Literal["fusion", "native"] = "fusion"
    retrieval_top_k: int = 6
    retrieval_timeout_s: float = 15.0

    # ---- Tools / MCP ----
    tool_timeout_s: float = 20.0
    mcp_transport: Literal["stdio", "http"] = "stdio"
    mcp_url: str = "http://localhost:8765/mcp"
    max_tool_iterations: int = 3

    # ---- Recursive Language Model (RLM) research agent ----
    rlm_max_depth: int = 2
    rlm_max_steps: int = 2
    rlm_batch_size: int = 4
    rlm_context_chars: int = 6000  # above this a sub-LM call is split recursively
    rlm_max_searches: int = 12
    rlm_max_llm_calls: int = 40
    rlm_concurrency: int = 4
    rlm_plan_timeout_s: float = 90.0

    # ---- Memory ----
    memory_recent_turns: int = 4
    memory_compress_after_turns: int = 8
    memory_recall_k: int = 3

    # ---- Security / limits ----
    rate_limits: dict[str, RateLimitConfig] = {
        "viewer": RateLimitConfig(capacity=5, refill_per_sec=0.1),
        "analyst": RateLimitConfig(capacity=10, refill_per_sec=0.2),
        "administrator": RateLimitConfig(capacity=20, refill_per_sec=0.5),
    }
    tool_rate_limit: RateLimitConfig = RateLimitConfig(capacity=40, refill_per_sec=2.0)
    max_message_chars: int = 4000
    injection_block_threshold: float = 0.8
    turn_timeout_s: float = 180.0

    # ---- Paths / misc ----
    data_dir: Path = ROOT / "data"
    today_override: str = ""
    api_url: str = "http://localhost:8000"

    # ---- derived ----
    @property
    def docs_dir(self) -> Path:
        return self.data_dir / "sample_docs"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def mcp_data_dir(self) -> Path:
        return self.data_dir / "mcp_data"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.openai_api_key:
            return "openai"
        if self.azure_openai_api_key and self.azure_openai_endpoint:
            return "azure"
        return "mock"

    @property
    def resolved_vector_backend(self) -> str:
        if self.vector_backend != "auto":
            return self.vector_backend
        return "pinecone" if self.pinecone_api_key else "local"

    @property
    def effective_embedding_dim(self) -> int:
        return MOCK_EMBEDDING_DIM if self.resolved_llm_provider == "mock" else self.embedding_dim

    def today(self) -> date:
        return date.fromisoformat(self.today_override) if self.today_override else date.today()


@lru_cache
def get_settings() -> Settings:
    return Settings()

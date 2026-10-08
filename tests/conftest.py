import asyncio
import shutil
from pathlib import Path

import pytest

from app.config import ROOT, get_settings


@pytest.fixture(scope="session")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    shutil.copytree(ROOT / "data" / "sample_docs", d / "sample_docs")
    shutil.copytree(ROOT / "data" / "mcp_data", d / "mcp_data")
    return d


@pytest.fixture(scope="session", autouse=True)
def offline_env(data_dir):
    """Every test runs fully offline: mock LLM, local vector store, throw-away data dir."""
    mp = pytest.MonkeyPatch()
    for k, v in {"LLM_PROVIDER": "mock", "VECTOR_BACKEND": "local", "OPENAI_API_KEY": "", "PINECONE_API_KEY": "",
                 "AZURE_OPENAI_API_KEY": "", "DATA_DIR": str(data_dir), "LANGSMITH_TRACING": "false",
                 "TODAY_OVERRIDE": "2026-10-08", "LOG_LEVEL": "ERROR"}.items():
        mp.setenv(k, v)
    get_settings.cache_clear()
    yield
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture(scope="session")
def container(offline_env):
    from app.services.container import Container
    return asyncio.run(Container.create(get_settings()))


@pytest.fixture
def users():
    from app.security.auth import authenticate
    return {n: authenticate(n, f"{n}123") for n in ("viewer", "analyst", "admin")}

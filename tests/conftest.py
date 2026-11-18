import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import reset_settings_cache  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_settings(monkeypatch):
    for name in (
        "OPENAI_API_KEY", "DEEPSEEK_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL",
        "CHAT_MODEL", "GITHUB_TOKEN", "GITHUB_REPO", "TOP_K", "SCORE_THRESHOLD",
        "BORDERLINE_LOW", "LLM_TIMEOUT_S",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def project_root() -> Path:
    return ROOT

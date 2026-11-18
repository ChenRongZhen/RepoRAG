"""集中读取 .env 与运行参数。

密钥只从环境/`.env` 进来，不写进代码；缺 key 时给出唯一一条中文报错。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_GITHUB_REPO = "he-yufeng/CoreCoder"


class ConfigError(RuntimeError):
    """配置缺失或非法。"""


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    chat_model: str
    github_token: str
    github_repo: str
    top_k: int = 3
    score_threshold: float = 0.72
    borderline_low: float = 0.45
    llm_timeout_s: float = 30.0
    github_timeout_s: tuple[float, float] = (3.05, 8.0)

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key.strip())

    def validate_for_llm(self) -> None:
        if not self.llm_configured:
            raise ConfigError(
                "缺少大模型 key：请在 RepoRAG/.env 里填写 OPENAI_API_KEY"
                "（或 DEEPSEEK_API_KEY），再重试。"
            )
        if not self.base_url.strip():
            raise ConfigError("缺少 OPENAI_BASE_URL，例如 https://api.deepseek.com")


def _number(name: str, default: float, cast=float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} 不是合法数字：{raw!r}") from exc


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """读一次 .env 并组装 Settings；已有环境变量优先。"""
    load_dotenv(ENV_FILE, override=False)

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY") or ""
    base_url = (
        os.getenv("OPENAI_BASE_URL")
        or os.getenv("DEEPSEEK_BASE_URL")
        or DEFAULT_BASE_URL
    )
    top_k = _number("TOP_K", 3, int)
    threshold = _number("SCORE_THRESHOLD", 0.72)
    low = _number("BORDERLINE_LOW", 0.45)
    if top_k <= 0:
        raise ConfigError("TOP_K 必须大于 0")
    if not 0 <= low <= threshold <= 1:
        raise ConfigError("需要满足 0 <= BORDERLINE_LOW <= SCORE_THRESHOLD <= 1")

    return Settings(
        api_key=api_key.strip(),
        base_url=base_url.strip(),
        chat_model=os.getenv("CHAT_MODEL", "deepseek-chat").strip(),
        github_token=os.getenv("GITHUB_TOKEN", "").strip(),
        github_repo=os.getenv("GITHUB_REPO", DEFAULT_GITHUB_REPO).strip(),
        top_k=top_k,
        score_threshold=threshold,
        borderline_low=low,
        llm_timeout_s=_number("LLM_TIMEOUT_S", 30.0),
    )


def reset_settings_cache() -> None:
    """测试里改完环境变量后调用。"""
    get_settings.cache_clear()

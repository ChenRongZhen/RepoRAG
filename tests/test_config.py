import pytest

from app.config import ConfigError, get_settings, reset_settings_cache


def test_missing_key_has_one_chinese_error(monkeypatch):
    # 不能 delenv：.env 里可能已经有真 key，delenv 之后 load_dotenv 会把它读回来。
    # 置空的环境变量优先级高于 .env，且 llm_configured 把空串当缺失。
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "")
    reset_settings_cache()
    with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
        get_settings().validate_for_llm()


def test_chat_model_does_not_collide_with_embedding_model(monkeypatch):
    monkeypatch.setenv("CHAT_MODEL", "deepseek-chat")
    settings = get_settings()
    assert settings.chat_model == "deepseek-chat"
    from app.embedding import MODEL_NAME
    assert MODEL_NAME == "BAAI/bge-small-zh-v1.5"


def test_invalid_thresholds(monkeypatch):
    monkeypatch.setenv("SCORE_THRESHOLD", "0.4")
    monkeypatch.setenv("BORDERLINE_LOW", "0.6")
    reset_settings_cache()
    with pytest.raises(ConfigError, match="BORDERLINE_LOW"):
        get_settings()


def test_default_base_url():
    assert get_settings().base_url

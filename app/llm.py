"""ChatOpenAI 工厂与结构化输出兜底。"""

from __future__ import annotations

import json
import re

from langchain_core.messages import BaseMessage
from langchain_openai import ChatOpenAI

from app.config import Settings, get_settings

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def get_chat_model(settings: Settings | None = None) -> ChatOpenAI:
    settings = settings or get_settings()
    settings.validate_for_llm()
    return ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.api_key,
        base_url=settings.base_url,
        temperature=0,
        timeout=settings.llm_timeout_s,
        max_retries=0,
    )


def _messages(prompt_value) -> list[BaseMessage]:
    return prompt_value.to_messages()


def structured_invoke(llm, schema, prompt_value):
    """按 function_calling → json_mode → 文本解析 依次尝试。

    DeepSeek 不支持 strict json_schema；前两种都失败时，从纯文本里抠 JSON。
    """
    messages = _messages(prompt_value)
    errors: list[Exception] = []

    for method in ("function_calling", "json_mode"):
        try:
            return llm.with_structured_output(schema, method=method).invoke(messages)
        except Exception as exc:  # noqa: BLE001 - provider 差异都走下一档
            errors.append(exc)

    try:
        reply = llm.invoke(messages)
        match = _JSON_BLOCK.search(str(reply.content))
        if match:
            return schema.model_validate(json.loads(match.group(0)))
        errors.append(ValueError("回复里没有 JSON"))
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)

    raise errors[-1]

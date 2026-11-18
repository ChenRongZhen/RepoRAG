"""离线演示用的假模型与假 Issue 检索（不联网、不花钱）。"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage

from app.prompts import REFUSAL_TEXT
from app.schemas import IssueHit, IssueSearchResult, RelevanceDecision

_DOC_ID = re.compile(r'<doc id="(D\d+)"')
_INSUFFICIENT = ("向量数据库", "Docker", "Kubernetes", "微调", "Postgres", "登录", "计费", "Web 界面")


def _text(messages) -> str:
    return "\n".join(str(getattr(message, "content", message)) for message in messages)


class _FakeStructured:
    def __init__(self, owner):
        self.owner = owner

    def invoke(self, messages):
        question = _text(messages)
        want_github = any(marker in question for marker in _INSUFFICIENT)
        return RelevanceDecision(
            enough_context=not want_github,
            reason="fake: 命中不足关键词" if want_github else "fake: 视为充足",
            github_query=question[:80],
        )


class FakeLLM:
    """确定性回复：有资料就引用第一条，否则拒答。"""

    def with_structured_output(self, schema, method=None):
        return _FakeStructured(self)

    def invoke(self, messages):
        text = _text(messages)
        ids = _DOC_ID.findall(text)
        if not ids:
            return AIMessage(content=REFUSAL_TEXT)
        return AIMessage(content=f"（离线假回答）资料中的相关说明见 [{ids[0]}]。")


class FakeIssueSearch:
    def __call__(self, question: str):
        return IssueSearchResult(
            status="ok",
            issues=[
                IssueHit(
                    number=1,
                    title=f"关于「{question[:30]}」的讨论",
                    state="open",
                    url="https://example.com/issues/1",
                    body="（离线假 Issue）",
                )
            ],
        )

"""跨模块共享的 Pydantic 结构。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RelevanceDecision(BaseModel):
    """相关性判定的结构化结果。"""

    enough_context: bool = Field(description="检索片段是否足以回答该问题")
    reason: str = Field(default="", description="一句话理由")
    github_query: str = Field(default="", description="不足时用于搜 Issue 的关键词")


class RagAnswer(BaseModel):
    """生成答案的结构化结果。"""

    answer: str


class IssueHit(BaseModel):
    """一条 GitHub Issue（只保留展示需要的字段）。"""

    number: int
    title: str
    state: str = ""
    url: str = ""
    body: str = ""
    updated_at: str = ""


IssueStatus = Literal["ok", "empty", "unavailable", "rate_limited", "error"]


class IssueSearchResult(BaseModel):
    """Issue 检索结果；失败也是一种结果，不抛异常。"""

    status: IssueStatus
    issues: list[IssueHit] = Field(default_factory=list)
    detail: str = ""

    @property
    def usable(self) -> bool:
        return self.status == "ok" and bool(self.issues)

"""GitHub Search Issues：只做补充检索，任何失败都返回状态而不是抛异常。"""

from __future__ import annotations

import re

import requests

from app.config import Settings, get_settings
from app.schemas import IssueHit, IssueSearchResult

API_URL = "https://api.github.com/search/issues"
MAX_QUERY_CHARS = 200
MAX_BODY_CHARS = 600

_STOPWORDS = {
    "的", "了", "是", "在", "和", "与", "怎么", "如何", "为什么", "什么", "哪些", "哪个",
    "corecoder", "the", "a", "an", "is", "are", "how", "why", "what", "which", "does", "do",
}
_MARKUP = re.compile(r"```.*?```|`[^`]*`|[*#>`|_\[\]()]", re.DOTALL)
_SPLIT = re.compile(r"[\s，。；：、！？,.!?;:]+")


def build_query(question: str, repo: str) -> str:
    """把问题压成 GitHub 搜索词；永远带上 repo 限制。

    GitHub 对中文分词很差，整句中文经常 0 命中；因此有 ASCII 词时只用 ASCII 词，
    没有时才退回中文词。关键词越少越好，多个词之间是 AND。
    """
    cleaned = _MARKUP.sub(" ", question)
    ascii_words: list[str] = []
    cjk_words: list[str] = []
    for word in _SPLIT.split(cleaned):
        word = word.lower()
        if word in _STOPWORDS:
            continue
        # 中文没有空格，停用词可能粘在短语里；ASCII 单词按整词判断，避免把 agent 削成 gent
        for stopword in sorted(_STOPWORDS, key=len, reverse=True):
            if not stopword.isascii():
                word = word.replace(stopword, "")
        word = word.strip()
        if len(word) < 2:
            continue
        if word.isascii():
            ascii_words.append(word)
        else:
            cjk_words.append(word)
    words = ascii_words if ascii_words else cjk_words
    limit = 4 if ascii_words else 3
    joined = " ".join(words[:limit])[:MAX_QUERY_CHARS].strip()
    if not joined:
        joined = question.strip()[:60]
    return f"repo:{repo} is:issue {joined} in:title,body"


def _parse(payload: dict) -> IssueSearchResult:
    issues: list[IssueHit] = []
    seen: set[int] = set()
    for item in payload.get("items", []):
        if "pull_request" in item or item.get("number") in seen:
            continue
        seen.add(item["number"])
        issues.append(
            IssueHit(
                number=item["number"],
                title=item.get("title", ""),
                state=item.get("state", ""),
                url=item.get("html_url", ""),
                body=(item.get("body") or "")[:MAX_BODY_CHARS],
                updated_at=item.get("updated_at", ""),
            )
        )
    if not issues:
        return IssueSearchResult(status="empty", detail="没有匹配的 Issue")
    return IssueSearchResult(status="ok", issues=issues)


def search_issues(
    question: str,
    settings: Settings | None = None,
    session: requests.Session | None = None,
    max_results: int = 5,
) -> IssueSearchResult:
    settings = settings or get_settings()
    session = session or requests.Session()

    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "RepoRAG/0.1",
    }
    if settings.github_token:
        headers["Authorization"] = f"Bearer {settings.github_token}"

    params = {
        "q": build_query(question, settings.github_repo),
        "sort": "updated",
        "order": "desc",
        "per_page": max(1, min(max_results, 5)),
    }

    for attempt in range(2):
        try:
            response = session.get(
                API_URL, params=params, headers=headers, timeout=settings.github_timeout_s
            )
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 0:
                continue
            return IssueSearchResult(status="unavailable", detail="GitHub 连接超时")
        except Exception as exc:  # noqa: BLE001
            return IssueSearchResult(status="error", detail=str(exc))

        if response.status_code == 200:
            try:
                return _parse(response.json())
            except ValueError:
                return IssueSearchResult(status="error", detail="GitHub 返回的不是 JSON")

        if response.status_code in (502, 503) and attempt == 0:
            continue

        if response.status_code == 401:
            return IssueSearchResult(status="unavailable", detail="GitHub token 无效")
        if response.status_code == 422:
            return IssueSearchResult(status="empty", detail="搜索词被 GitHub 拒绝")
        if response.status_code in (403, 429):
            remaining = response.headers.get("X-RateLimit-Remaining", "")
            retry_after = response.headers.get("Retry-After", "")
            if attempt == 0 and not remaining.isdigit() and retry_after.isdigit() and int(retry_after) <= 2:
                continue
            if remaining == "0" or response.status_code == 429:
                return IssueSearchResult(status="rate_limited", detail="GitHub 限流")
            return IssueSearchResult(status="unavailable", detail="GitHub 拒绝访问")
        return IssueSearchResult(status="error", detail=f"GitHub 状态码 {response.status_code}")

    return IssueSearchResult(status="unavailable", detail="GitHub 重试后仍失败")

import requests

from app.config import get_settings
from app.github_issues import build_query, search_issues

OK_PAYLOAD = {
    "items": [
        {"number": 2, "title": "真 Issue", "state": "open",
         "html_url": "https://example.com/2", "body": "x" * 900, "updated_at": "now"},
        {"number": 2, "title": "重复", "state": "open", "html_url": "https://example.com/2"},
        {"number": 3, "title": "PR", "pull_request": {}, "html_url": "https://example.com/3"},
    ]
}


class FakeResponse:
    def __init__(self, status=200, payload=None, headers=None):
        self.status_code = status
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("bad json")
        return self._payload


class FakeSession:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = 0

    def get(self, *args, **kwargs):
        self.calls += 1
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_ok_parses_dedupes_and_drops_prs():
    result = search_issues("崩溃", get_settings(), FakeSession(FakeResponse(payload=OK_PAYLOAD)))
    assert result.status == "ok"
    assert [issue.number for issue in result.issues] == [2]
    assert len(result.issues[0].body) == 600


def test_empty_payload():
    result = search_issues("q", get_settings(), FakeSession(FakeResponse(payload={"items": []})))
    assert result.status == "empty"


def test_rate_limit_with_zero_remaining():
    result = search_issues(
        "q", get_settings(),
        FakeSession(FakeResponse(status=403, headers={"X-RateLimit-Remaining": "0"})),
    )
    assert result.status == "rate_limited"


def test_retry_after_is_retried_once():
    session = FakeSession(
        FakeResponse(status=429, headers={"Retry-After": "1"}),
        FakeResponse(payload={"items": []}),
    )
    result = search_issues("q", get_settings(), session)
    assert result.status == "empty"
    assert session.calls == 2


def test_timeout_twice_is_unavailable():
    session = FakeSession(requests.Timeout(), requests.Timeout())
    result = search_issues("q", get_settings(), session)
    assert result.status == "unavailable"


def test_bad_json_is_error():
    result = search_issues("q", get_settings(), FakeSession(FakeResponse()))
    assert result.status == "error"


def test_401_is_unavailable():
    result = search_issues("q", get_settings(), FakeSession(FakeResponse(status=401)))
    assert result.status == "unavailable"


def test_422_is_empty():
    result = search_issues("q", get_settings(), FakeSession(FakeResponse(status=422)))
    assert result.status == "empty"


def test_502_is_retried():
    session = FakeSession(FakeResponse(status=502), FakeResponse(payload={"items": []}))
    result = search_issues("q", get_settings(), session)
    assert result.status == "empty"
    assert session.calls == 2


def test_query_prefers_ascii_and_caps_terms():
    query = build_query("Kimi 模型第一次工具调用后失败，reasoning_content 没回传", "o/r")
    terms = query.split(" in:title,body")[0].split()[2:]
    assert terms == ["kimi", "reasoning", "content"]


def test_query_falls_back_to_chinese_and_caps_at_three():
    query = build_query("上下文 压缩 阈值 孤儿 消息 怎么 处理", "o/r")
    assert "上下文" in query
    assert len(query.split(" in:title,body")[0].split()[2:]) <= 3


def test_query_is_scoped_and_bounded():
    query = build_query("CoreCoder 的 Agent 如何运行？" + "长" * 300, "o/r")
    assert query.startswith("repo:o/r is:issue")
    assert "如何" not in query
    assert len(query) <= 240
    assert "gent" not in query.split()

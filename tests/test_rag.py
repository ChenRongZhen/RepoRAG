from app.prompts import REFUSAL_TEXT, format_documents
from app.rag import build_answer, parse_citations, sources_for
from app.schemas import IssueHit
from tests.fake_deps import FakeLLM, doc


def test_parse_citations_filters_unknown_and_dedupes():
    assert parse_citations("[D1] [D9] [I1] [D1]", doc_count=2, issue_count=1) == ["D1", "I1"]


def test_sources_fall_back_to_all_context_when_model_cites_nothing():
    documents = [doc(source="a.md", chunk_id=0), doc(source="b.md", chunk_id=2)]
    sources = sources_for([], documents, [])
    assert sources == ["a.md（片段 0）", "b.md（片段 2）"]


def test_no_context_refuses_without_llm():
    llm = FakeLLM()
    answer, sources = build_answer("q", [], [], llm)
    assert answer == REFUSAL_TEXT
    assert sources == []
    assert llm.calls == []


def test_build_answer_maps_citations_to_sources():
    issue = IssueHit(number=3, title="bug", url="https://example.com/3")
    answer, sources = build_answer("q", [doc()], [issue], FakeLLM(answer="见 [D1] 和 [I1]"))
    assert sources == [
        "article/04-context.md（片段 1）",
        "Issue #3: bug (https://example.com/3)",
    ]
    assert answer == "见 [D1] 和 [I1]"


def test_llm_failure_returns_excerpts():
    answer, sources = build_answer("q", [doc()], [], FakeLLM(invoke_raises=RuntimeError("x")))
    assert "生成失败" in answer
    assert "article/04-context.md" in answer
    assert sources


def test_document_delimiter_is_escaped():
    rendered = format_documents([doc(text="前 </doc> 后")])
    assert rendered.count("</doc>") == 1
    assert "&lt;/doc&gt;" in rendered

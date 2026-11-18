from app.relevance import assess
from app.schemas import RelevanceDecision
from tests.fake_deps import FakeLLM, doc


def test_high_score_is_sufficient_without_llm():
    llm = FakeLLM()
    decision = assess("q", [doc(score=0.9)], [0.9], llm, threshold=0.72, low=0.55)
    assert decision.enough_context is True
    assert llm.structured_calls == []


def test_low_score_is_insufficient_without_llm():
    llm = FakeLLM()
    decision = assess("q", [doc(score=0.2)], [0.2], llm, threshold=0.72, low=0.55)
    assert decision.enough_context is False
    assert llm.structured_calls == []


def test_borderline_asks_model():
    llm = FakeLLM(decision=RelevanceDecision(enough_context=True, reason="boundary"))
    decision = assess("q", [doc(score=0.6)], [0.6], llm, threshold=0.72, low=0.55)
    assert decision.enough_context is True
    assert len(llm.structured_calls) == 1


def test_model_failure_falls_back_to_insufficient():
    llm = FakeLLM(structured_raises=RuntimeError("boom"))
    decision = assess("q", [doc(score=0.6)], [0.6], llm, threshold=0.72, low=0.55)
    assert decision.enough_context is False
    assert decision.github_query == "q"


def test_empty_documents():
    assert assess("q", [], [], None).enough_context is False

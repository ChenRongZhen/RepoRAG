"""LangGraph 工作流：检索 → 判定 → 生成 / 查 Issue / 拒答。"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from app.github_issues import search_issues
from app.llm import get_chat_model
from app.prompts import GITHUB_UNAVAILABLE_NOTE, REFUSAL_TEXT
from app.rag import build_answer
from app.relevance import assess
from app.schemas import IssueHit, RelevanceDecision


class AgentState(TypedDict, total=False):
    question: str
    documents: list
    doc_scores: list[float]
    relevance: RelevanceDecision | None
    enough_context: bool
    github_issues: list[IssueHit]
    github_status: str
    answer: str
    sources: list[str]
    refusal_reason: str
    error: str | None
    trace: Annotated[list[str], operator.add]


def build_graph(*, retriever, llm, issue_search=None, threshold=0.72, low=0.45):
    """所有依赖注入，tests 可零网络替换。"""

    def retrieve_docs(state: AgentState) -> dict:
        question = state.get("question", "").strip()
        if not question:
            return {"documents": [], "doc_scores": [], "error": "empty_question",
                    "trace": ["retrieve:empty"]}
        try:
            documents = retriever.invoke(question)
        except Exception as exc:  # noqa: BLE001
            return {"documents": [], "doc_scores": [], "error": str(exc),
                    "trace": [f"retrieve:error={exc.__class__.__name__}"]}
        scores = [float(document.metadata.get("score", 0.0)) for document in documents]
        return {"documents": documents, "doc_scores": scores,
                "trace": [f"retrieve:n={len(documents)}"]}

    def check_relevance(state: AgentState) -> dict:
        if state.get("error") == "empty_question":
            return {"enough_context": False, "trace": ["relevance:skip"]}
        decision = assess(
            state.get("question", ""),
            state.get("documents", []),
            state.get("doc_scores", []),
            llm,
            threshold=threshold,
            low=low,
        )
        route = "generate_answer" if decision.enough_context else "github"
        return {
            "relevance": decision,
            "enough_context": decision.enough_context,
            "trace": [f"relevance:enough={decision.enough_context}:{route}"],
        }

    def search_github(state: AgentState) -> dict:
        if issue_search is None:
            return {"github_issues": [], "github_status": "disabled",
                    "trace": ["github:disabled"]}
        question = state.get("question", "")
        decision = state.get("relevance")
        query = (decision.github_query if decision and decision.github_query else question)
        try:
            result = issue_search(query)
        except Exception as exc:  # noqa: BLE001
            return {"github_issues": [], "github_status": "error",
                    "trace": [f"github:error={exc.__class__.__name__}"]}
        return {
            "github_issues": result.issues[:5],
            "github_status": result.status,
            "trace": [f"github:{result.status}:n={len(result.issues)}"],
        }

    def generate_answer(state: AgentState) -> dict:
        answer, sources = build_answer(
            state.get("question", ""),
            state.get("documents", []),
            state.get("github_issues", []),
            llm,
        )
        if state.get("github_status") not in (None, "ok") and answer.startswith(REFUSAL_TEXT):
            answer += "\n" + GITHUB_UNAVAILABLE_NOTE
        return {"answer": answer, "sources": sources, "trace": ["answer"]}

    def refuse(state: AgentState) -> dict:
        reason = "empty_question" if state.get("error") == "empty_question" else "no_context"
        answer = "问题不能为空。" if reason == "empty_question" else REFUSAL_TEXT
        status = state.get("github_status")
        if status in ("disabled", "unavailable", "rate_limited", "error"):
            answer += "\n" + GITHUB_UNAVAILABLE_NOTE
        return {"answer": answer, "sources": [], "refusal_reason": reason,
                "trace": [f"refuse:{reason}"]}

    def route_after_relevance(state: AgentState) -> str:
        if state.get("error") == "empty_question":
            return "refuse"
        if state.get("enough_context"):
            return "generate_answer"
        return "search_github"

    def route_after_github(state: AgentState) -> str:
        return "generate_answer" if state.get("github_issues") else "refuse"

    builder = StateGraph(AgentState)
    builder.add_node("retrieve_docs", retrieve_docs)
    builder.add_node("check_relevance", check_relevance)
    builder.add_node("search_github", search_github)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("refuse", refuse)

    builder.add_edge(START, "retrieve_docs")
    builder.add_edge("retrieve_docs", "check_relevance")
    builder.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {"generate_answer": "generate_answer", "search_github": "search_github", "refuse": "refuse"},
    )
    builder.add_conditional_edges(
        "search_github",
        route_after_github,
        {"generate_answer": "generate_answer", "refuse": "refuse"},
    )
    builder.add_edge("generate_answer", END)
    builder.add_edge("refuse", END)
    return builder.compile()


def default_graph(*, no_github: bool = False, fake: bool = False, top_k: int = 3):
    """按配置组装真实依赖；fake=True 时完全离线。"""
    from app.config import get_settings
    from app.fakes import FakeIssueSearch, FakeLLM
    from app.retrieve import BgeRetriever

    settings = get_settings()
    llm = FakeLLM() if fake else get_chat_model(settings)
    issue_search = None if no_github else (FakeIssueSearch() if fake else search_issues)
    return build_graph(
        retriever=BgeRetriever(top_k=top_k),
        llm=llm,
        issue_search=issue_search,
        threshold=settings.score_threshold,
        low=settings.borderline_low,
    )


def run(question: str, graph=None) -> AgentState:
    graph = graph or default_graph()
    return graph.invoke({"question": question, "trace": []})

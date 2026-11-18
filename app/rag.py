"""回答生成与引用映射；不依赖 LangGraph，便于先单独验收。"""

from __future__ import annotations

import re

from app.prompts import ANSWER_PROMPT, REFUSAL_TEXT, format_documents, format_issues
from app.schemas import IssueHit

_CITATION = re.compile(r"\[(D\d+|I\d+)\]")
MAX_FALLBACK_EXCERPT = 300


def parse_citations(answer: str, doc_count: int, issue_count: int) -> list[str]:
    """只保留真实存在的引用 id，按出现顺序去重。"""
    allowed = {f"D{i}" for i in range(1, doc_count + 1)}
    allowed |= {f"I{i}" for i in range(1, issue_count + 1)}
    seen: list[str] = []
    for citation in _CITATION.findall(answer or ""):
        if citation in allowed and citation not in seen:
            seen.append(citation)
    return seen


def sources_for(citations: list[str], documents, issues: list[IssueHit]) -> list[str]:
    """把引用 id 映射成可读来源；模型没有标注时回退到全部上下文来源。"""
    if not citations:
        citations = [f"D{i}" for i in range(1, len(documents) + 1)]
        citations += [f"I{i}" for i in range(1, len(issues) + 1)]

    sources: list[str] = []
    for citation in citations:
        kind, number = citation[0], int(citation[1:])
        if kind == "D" and 1 <= number <= len(documents):
            metadata = documents[number - 1].metadata
            sources.append(f"{metadata.get('source', '?')}（片段 {metadata.get('chunk_id', '?')}）")
        elif kind == "I" and 1 <= number <= len(issues):
            issue = issues[number - 1]
            sources.append(f"Issue #{issue.number}: {issue.title} ({issue.url})")
    return sources


def build_answer(question: str, documents, issues, llm) -> tuple[str, list[str]]:
    """调用模型生成答案并返回 (answer, sources)。异常不抛出。"""
    if not documents and not issues:
        return REFUSAL_TEXT, []

    try:
        message = llm.invoke(
            ANSWER_PROMPT.invoke({
                "question": question,
                "context": format_documents(documents),
                "issues": format_issues(issues),
            })
        )
        answer = (message.content or "").strip()
        if not answer:
            raise ValueError("模型返回空答案")
    except Exception as exc:  # noqa: BLE001
        excerpts = "\n".join(
            f"- {document.metadata.get('source', '?')}: "
            f"{document.page_content.replace(chr(10), ' ')[:MAX_FALLBACK_EXCERPT]}"
            for document in documents[:3]
        )
        answer = f"生成失败（{exc.__class__.__name__}）。可先看这些原文：\n{excerpts}"
        return answer, sources_for([], documents, issues)

    citations = parse_citations(answer, len(documents), len(issues))
    if answer.startswith(REFUSAL_TEXT) and not citations:
        return answer, []
    return answer, sources_for(citations, documents, issues)


def answer_question(question: str, retriever, llm, issue_search=None, *, top_k: int = 3):
    """阶段二的直连版本：检索 → 判定 → 必要时查 Issue → 作答。"""
    from app.relevance import assess

    documents = retriever.invoke(question) if question.strip() else []
    scores = [document.metadata.get("score", 0.0) for document in documents]
    decision = assess(question, documents, scores, llm)
    issues = []
    if not decision.enough_context and issue_search is not None:
        result = issue_search(question)
        if result.usable:
            issues = result.issues[:3]
    return build_answer(question, documents, issues, llm)

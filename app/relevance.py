"""相关性判定：分数先筛，临界区间才问模型。"""

from __future__ import annotations

from app.prompts import RELEVANCE_PROMPT, format_documents
from app.schemas import RelevanceDecision


def assess(
    question: str,
    documents,
    scores: list[float],
    llm=None,
    *,
    threshold: float = 0.72,
    low: float = 0.45,
) -> RelevanceDecision:
    """返回是否足以作答；任何异常都退化成保守结论。"""
    if not documents:
        return RelevanceDecision(enough_context=False, reason="没有检索到片段")

    best = max(scores) if scores else 0.0
    if best < low:
        return RelevanceDecision(enough_context=False, reason=f"最高分 {best:.3f} 低于下限")
    if best >= threshold:
        return RelevanceDecision(enough_context=True, reason=f"最高分 {best:.3f} 达到阈值")

    if llm is None:
        return RelevanceDecision(enough_context=False, reason="临界分数且没有判定模型")

    try:
        from app.llm import structured_invoke

        decision = structured_invoke(
            llm,
            RelevanceDecision,
            RELEVANCE_PROMPT.invoke({
                "question": question,
                "context": format_documents(documents[:3]),
            }),
        )
        if not decision.github_query.strip():
            decision.github_query = question
        return decision
    except Exception as exc:  # noqa: BLE001 - 判定失败不能拖垮图
        return RelevanceDecision(
            enough_context=False,
            reason=f"判定模型失败（{exc.__class__.__name__}），按分数保守处理",
            github_query=question,
        )

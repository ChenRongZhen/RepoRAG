"""离线检索评测 + 完整回答评测。

用法（项目根目录）：
    python -m app.eval --retrieval-only     只跑检索指标，不花模型调用
    python -m app.eval                      再跑完整回答、拒答与引用指标
    python -m app.eval --strict             低于目标阈值时退出码为 1
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.prompts import REFUSAL_TEXT
from app.retrieve import retrieve_with_scores

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = PROJECT_ROOT / "tests" / "questions.json"
REPORT_PATH = PROJECT_ROOT / "eval" / "report.json"
REPORT_MD_PATH = PROJECT_ROOT / "eval" / "report.md"

TOP_K = 5
TARGET_TOP1 = 0.50
TARGET_TOP3 = 0.70
TARGET_TOP5 = 0.80
TARGET_REFUSAL_PRECISION = 0.80
TARGET_CITATION_COVERAGE = 0.90

CITATION = re.compile(r"\[(D|I)(\d+)\]")


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank_of_first_hit(sources: list[str], expected: list[str]) -> int | None:
    """第一个命中期望来源的 1-based 名次；未命中返回 None。"""
    wanted = set(expected)
    for rank, source in enumerate(sources, start=1):
        if source in wanted:
            return rank
    return None


def retrieve_sources(question: str, top_k: int = TOP_K) -> list[str]:
    return [document.metadata.get("source", "") for document in retrieve_with_scores(question, top_k)]


def run_retrieval_eval(items: list[dict]) -> dict:
    rows = []
    for item in items:
        if item["category"] != "doc":
            continue
        sources = retrieve_sources(item["question"])
        rows.append({
            "id": item["id"],
            "question": item["question"],
            "expected": item["expected_sources"],
            "retrieved": sources,
            "rank": rank_of_first_hit(sources, item["expected_sources"]),
        })
    total = len(rows)
    hits = {k: sum(1 for row in rows if row["rank"] is not None and row["rank"] <= k) for k in (1, 3, 5)}
    return {
        "total": total,
        "top1": hits[1] / total if total else 0.0,
        "top3": hits[3] / total if total else 0.0,
        "top5": hits[5] / total if total else 0.0,
        "rows": rows,
    }


def _answer_one(graph, item: dict) -> dict:
    state = graph.invoke({"question": item["question"], "trace": []})
    answer = state.get("answer", "")
    sources = state.get("sources", [])
    document_sources = {s for s in sources if not s.startswith("http")}
    cited = {f"D{number}" for _, number in CITATION.findall(answer)}
    return {
        "id": item["id"],
        "category": item["category"],
        "question": item["question"],
        "answer": answer,
        "refused": answer.startswith(REFUSAL_TEXT) or answer.startswith("问题不能为空"),
        "expect_refusal": item["expect_refusal"],
        "sources": sources,
        "citation_count": len(cited),
        "citation_ok": bool(cited) or not document_sources,
        "trace": state.get("trace", []),
    }


def run_answer_eval(items: list[dict], graph) -> dict:
    rows = [_answer_one(graph, item) for item in items]
    refusals = [row for row in rows if row["category"] == "refusal"]
    answerable = [row for row in rows if not row["expect_refusal"]]
    correct_refusals = sum(1 for row in refusals if row["refused"])
    false_refusals = sum(1 for row in answerable if row["refused"])
    covered = sum(1 for row in answerable if not row["refused"] and row["citation_ok"])
    return {
        "total": len(rows),
        "refusal_precision": correct_refusals / len(refusals) if refusals else 0.0,
        "false_refusal_rate": false_refusals / len(answerable) if answerable else 0.0,
        "citation_coverage": covered / len(answerable) if answerable else 0.0,
        "rows": rows,
    }


def strict_failures(report: dict) -> list[str]:
    retrieval = report.get("retrieval") or {}
    answer = report.get("answer") or {}
    failures = []
    if (retrieval.get("total") or 0) < 24:
        failures.append(f"文档题不足 24 条（实际 {retrieval.get('total', 0)}）")
    if retrieval.get("top3", 0) < TARGET_TOP3:
        failures.append(f"Top-3 {retrieval.get('top3', 0):.1%} < {TARGET_TOP3:.0%}")
    if retrieval.get("top5", 0) < TARGET_TOP5:
        failures.append(f"Top-5 {retrieval.get('top5', 0):.1%} < {TARGET_TOP5:.0%}")
    if answer:
        if answer.get("refusal_precision", 0) < TARGET_REFUSAL_PRECISION:
            failures.append(
                f"拒答精确率 {answer.get('refusal_precision', 0):.1%} < {TARGET_REFUSAL_PRECISION:.0%}"
            )
        if answer.get("citation_coverage", 0) < TARGET_CITATION_COVERAGE:
            failures.append(
                f"引用覆盖率 {answer.get('citation_coverage', 0):.1%} < {TARGET_CITATION_COVERAGE:.0%}"
            )
    return failures


def write_reports(report: dict) -> None:
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    retrieval = report.get("retrieval") or {}
    lines = ["# 评测报告", "", f"- 生成时间：{report['generated_at']}", ""]
    if retrieval:
        lines += [
            "## 检索（仅文档题）",
            "",
            f"- 题目数：{retrieval['total']}",
            f"- Top-1：{retrieval['top1']:.1%}",
            f"- Top-3：{retrieval['top3']:.1%}",
            f"- Top-5：{retrieval['top5']:.1%}",
            "",
            "| 题号 | 首个命中名次 | 问题 |",
            "|---|---|---|",
        ]
        for row in retrieval["rows"]:
            lines.append(f"| {row['id']} | {row['rank'] or '未命中'} | {row['question']} |")
        lines.append("")
    answer = report.get("answer")
    if answer:
        lines += [
            "## 回答",
            "",
            f"- 拒答精确率：{answer['refusal_precision']:.1%}",
            f"- 误拒率：{answer['false_refusal_rate']:.1%}",
            f"- 引用覆盖率：{answer['citation_coverage']:.1%}",
            "",
        ]
    REPORT_MD_PATH.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RepoRAG 评测")
    parser.add_argument("--retrieval-only", action="store_true", help="只跑本地检索指标")
    parser.add_argument("--strict", action="store_true", help="低于目标阈值时失败")
    parser.add_argument("--no-github", action="store_true", help="回答评测时关闭 GitHub 分支")
    parser.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    args = parser.parse_args(argv)

    from datetime import datetime

    items = load_questions(args.questions)
    report: dict = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "retrieval": run_retrieval_eval(items),
    }

    if not args.retrieval_only:
        from app.graph import default_graph

        report["answer"] = run_answer_eval(items, default_graph(no_github=args.no_github))

    write_reports(report)
    retrieval = report["retrieval"]
    print(f"检索题 {retrieval['total']} 条：Top-1 {retrieval['top1']:.1%}，"
          f"Top-3 {retrieval['top3']:.1%}，Top-5 {retrieval['top5']:.1%}")
    if report.get("answer"):
        answer = report["answer"]
        print(f"拒答精确率 {answer['refusal_precision']:.1%}，误拒率 {answer['false_refusal_rate']:.1%}，"
              f"引用覆盖率 {answer['citation_coverage']:.1%}")
    print(f"报告：{REPORT_PATH}")

    if args.strict:
        failures = strict_failures(report)
        for failure in failures:
            print("未达标：" + failure)
        if failures:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

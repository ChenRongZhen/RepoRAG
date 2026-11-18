"""RepoRAG 命令行入口：一次性提问或交互式问答。"""

from __future__ import annotations

import argparse
import json
import sys

from app.config import ConfigError, get_settings
from app.graph import default_graph


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RepoRAG：开源项目文档问答 Agent")
    parser.add_argument("question", nargs="?", help="一次性提问；省略则进入交互模式")
    parser.add_argument("--fake", action="store_true", help="离线假模型，不联网不花钱")
    parser.add_argument("--no-github", action="store_true", help="关闭 GitHub Issues 分支")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出")
    parser.add_argument("--top-k", type=int, default=None, help="检索片段数（默认 3）")
    parser.add_argument("--debug", action="store_true", help="显示执行轨迹")
    return parser.parse_args()


def _render(state: dict, as_json: bool, debug: bool) -> None:
    if as_json:
        print(json.dumps(
            {
                "answer": state.get("answer", ""),
                "sources": state.get("sources", []),
                "refusal_reason": state.get("refusal_reason", ""),
                "trace": state.get("trace", []),
            },
            ensure_ascii=False,
        ))
        return

    print("\n" + state.get("answer", ""))
    sources = state.get("sources", [])
    if sources:
        print("\n来源：")
        for source in sources:
            print(f"- {source}")
    if debug:
        print("\n[轨迹] " + " → ".join(state.get("trace", [])))


def _ask(graph, question: str, args) -> None:
    try:
        state = graph.invoke({"question": question, "trace": []})
    except KeyboardInterrupt:
        print("\n已中断。")
        return
    except Exception as exc:  # noqa: BLE001
        print(f"运行失败：{exc}", file=sys.stderr)
        return
    _render(state, args.json, args.debug)


def main() -> int:
    args = _parse_args()

    if not args.fake:
        try:
            get_settings().validate_for_llm()
        except ConfigError as exc:
            print(exc, file=sys.stderr)
            return 2

    try:
        graph = default_graph(no_github=args.no_github, fake=args.fake,
                              top_k=args.top_k or get_settings().top_k)
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 2

    if args.question is not None:
        _ask(graph, args.question, args)
        return 0

    print("RepoRAG：CoreCoder 文档检索问答")
    print("输入 quit 或 exit 退出。")
    while True:
        try:
            question = input("\n请输入问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            return 0
        if question.lower() in {"quit", "exit"}:
            print("再见！")
            return 0
        if not question:
            print("问题不能为空。")
            continue
        _ask(graph, question, args)


if __name__ == "__main__":
    raise SystemExit(main())

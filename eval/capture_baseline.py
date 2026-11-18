"""在改造前记录旧检索器的 Top-3，作为回归基准。

用法（项目根目录）：python -m eval.capture_baseline
"""

import json
from pathlib import Path

from app.retrieve import search

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = PROJECT_ROOT / "tests" / "questions.json"
OUTPUT = Path(__file__).resolve().parent / "baseline_top3.json"


def main() -> None:
    items = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    rows = []
    for item in items:
        question = item["question"]
        results = search(question, top_k=3)
        rows.append({
            "question": question,
            "top3": [
                {
                    "source": result.chunk.source,
                    "chunk_id": result.chunk.chunk_id,
                    "score": result.score,
                }
                for result in results
            ],
        })
    OUTPUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已写入 {OUTPUT}（{len(rows)} 条）")


if __name__ == "__main__":
    main()

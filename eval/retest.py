"""开源前的人工回归：跑固定问题集，记录输出、来源、耗时与路由。"""

import json
import subprocess
import sys
import time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]

QUESTIONS = [
    ("doc", "上下文压缩分哪三层，阈值是多少？"),
    ("doc", "edit_file 为什么要求唯一匹配？"),
    ("doc", "子 agent 为什么不准再派孙 agent？"),
    ("doc", "会话名怎么做路径穿越防护？"),
    ("refuse", "CoreCoder 用的是哪个向量数据库？"),
    ("refuse", "怎么用 Docker 部署？"),
    ("refuse", "会话数据存在 Postgres 还是 Redis？"),
    ("refuse", "模型的微调脚本在哪？"),
    ("github", "edit_file 在 Windows 上会把行尾改成 CRLF 吗？"),
    ("github", "temperature=0 时 Kimi 请求报 400 怎么办？"),
    ("ambiguous", "上下文怎么处理？"),
    ("empty", ""),
]

rows = []
for kind, question in QUESTIONS:
    started = time.perf_counter()
    proc = subprocess.run(
        [PY, "main.py", "--json", "--no-github" if kind == "github" and False else "--json", question]
        if False else [PY, "main.py", "--json", question],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=180,
    )
    elapsed = time.perf_counter() - started
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception:
        payload = {"answer": proc.stdout.strip(), "sources": [], "trace": [], "raw": proc.stderr}
    rows.append({
        "kind": kind,
        "question": question,
        "seconds": round(elapsed, 2),
        "refused": payload.get("answer", "").startswith("根据现有资料无法回答"),
        "sources": payload.get("sources", []),
        "trace": payload.get("trace", []),
        "answer": payload.get("answer", ""),
        "exit": proc.returncode,
    })
    print(f"[{kind}] {elapsed:5.1f}s {question!r} -> refused={rows[-1]['refused']} sources={len(rows[-1]['sources'])}")

out = ROOT / "eval" / "retest.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
times = sorted(row["seconds"] for row in rows if row["kind"] != "empty")
print("median", times[len(times) // 2], "max", times[-1])
print("doc answered", sum(1 for r in rows if r["kind"] == "doc" and not r["refused"]), "/4")
print("refusals", sum(1 for r in rows if r["kind"] == "refuse" and r["refused"]), "/4")
print("github answered", sum(1 for r in rows if r["kind"] == "github" and not r["refused"]), "/4")

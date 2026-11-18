import json

from app.eval import load_questions, rank_of_first_hit, strict_failures


def test_rank_of_first_hit():
    assert rank_of_first_hit(["a.md", "b.md"], ["b.md"]) == 2
    assert rank_of_first_hit(["a.md"], ["b.md"]) is None
    assert rank_of_first_hit([], ["b.md"]) is None


def test_question_set_shape():
    items = load_questions()
    assert len(items) == 36
    assert len({item["id"] for item in items}) == 36
    categories = [item["category"] for item in items]
    assert categories.count("doc") == 24
    assert categories.count("refusal") == 8
    assert categories.count("github") == 4
    for item in items:
        assert item["question"].strip()
        assert isinstance(item["expect_refusal"], bool)
        if item["category"] == "doc":
            assert item["expected_sources"]


def test_strict_failures_reports_each_miss():
    report = {
        "retrieval": {"total": 24, "top3": 0.5, "top5": 0.6},
        "answer": {"refusal_precision": 0.5, "citation_coverage": 0.5},
    }
    failures = "\n".join(strict_failures(report))
    assert "Top-3" in failures
    assert "Top-5" in failures
    assert "拒答精确率" in failures
    assert "引用覆盖率" in failures


def test_strict_passes_healthy_report():
    report = {
        "retrieval": {"total": 24, "top3": 0.9, "top5": 1.0},
        "answer": {"refusal_precision": 1.0, "citation_coverage": 1.0},
    }
    assert strict_failures(report) == []


def test_report_json_is_written_with_utf8(tmp_path, monkeypatch):
    import app.eval as eval_module

    monkeypatch.setattr(eval_module, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(eval_module, "REPORT_MD_PATH", tmp_path / "report.md")
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps([{"id": "d1", "category": "doc", "question": "q",
                     "expected_sources": ["README_CN.md"], "expect_refusal": False}]),
        encoding="utf-8",
    )
    assert eval_module.main(["--retrieval-only", "--questions", str(questions)]) == 0
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert saved["retrieval"]["total"] == 1

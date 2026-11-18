import json
from pathlib import Path

import pytest

from app.retrieve import BgeRetriever, clear_index_cache, search

QUESTIONS = Path(__file__).resolve().parents[1] / "tests" / "questions.json"


@pytest.fixture(scope="module", autouse=True)
def _cache():
    clear_index_cache()
    yield
    clear_index_cache()


@pytest.fixture(scope="module")
def questions():
    return [item["question"] for item in json.loads(QUESTIONS.read_text(encoding="utf-8"))]


def test_adapter_matches_legacy_search(questions):
    retriever = BgeRetriever(top_k=3)
    for question in questions:
        legacy = [(r.chunk.source, r.chunk.chunk_id, r.score) for r in search(question, 3)]
        adapted = [
            (d.metadata["source"], d.metadata["chunk_id"], d.metadata["score"])
            for d in retriever.invoke(question)
        ]
        assert adapted == legacy, question

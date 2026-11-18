import joblib
import numpy as np
import pytest

from app import retrieve
from app.retrieve import clear_index_cache, load_index, retrieve_with_scores, search


@pytest.fixture(autouse=True)
def _clear_cache():
    clear_index_cache()
    yield
    clear_index_cache()


def test_search_returns_descending_scores():
    results = search("上下文压缩分几层？", top_k=3)
    assert len(results) == 3
    scores = [result.score for result in results]
    assert scores == sorted(scores, reverse=True)
    assert all(isinstance(score, float) for score in scores)


def test_empty_query_rejected():
    with pytest.raises(ValueError, match="问题不能为空"):
        search("  ")


def test_bad_top_k_rejected():
    with pytest.raises(ValueError, match="top_k"):
        search("q", top_k=0)


def test_retriever_documents_carry_metadata():
    documents = retrieve_with_scores("上下文压缩分几层？", top_k=2)
    assert all(document.metadata["kind"] == "doc" for document in documents)
    assert all("source" in document.metadata for document in documents)


def test_model_mismatch_guard(tmp_path):
    path = tmp_path / "index.joblib"
    joblib.dump({"model_name": "other", "embeddings": np.zeros((0, 2)), "chunks": []}, path)
    with pytest.raises(ValueError, match="模型"):
        load_index(path)


def test_missing_index_auto_builds(tmp_path, monkeypatch):
    path = tmp_path / "auto.joblib"

    def fake_build(index_path):
        joblib.dump(
            {"model_name": retrieve.MODEL_NAME, "embeddings": np.zeros((1, 2)),
             "chunks": [], "schema_version": 2},
            index_path,
        )
        return index_path

    monkeypatch.setattr(retrieve, "build_index", fake_build)
    embeddings, chunks = load_index(path)
    assert embeddings.shape == (1, 2)
    assert chunks == []

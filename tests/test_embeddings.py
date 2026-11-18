import numpy as np

from app import embeddings


def test_query_instruction_is_applied_once(monkeypatch):
    seen = {}

    def fake_query(text):
        seen["query"] = text
        return np.array([1.0, 0.0])

    monkeypatch.setattr(embeddings, "encode_query", fake_query)
    vector = embeddings.BGEEmbeddings().embed_query("上下文太长怎么办")
    assert seen["query"] == "上下文太长怎么办"
    assert vector == [1.0, 0.0]


def test_documents_do_not_get_instruction(monkeypatch):
    seen = {}

    def fake_docs(texts):
        seen["docs"] = texts
        return np.array([[1.0, 0.0], [0.0, 1.0]])

    monkeypatch.setattr(embeddings, "encode_documents", fake_docs)
    vectors = embeddings.BGEEmbeddings().embed_documents(["a", "b"])
    assert seen["docs"] == ["a", "b"]
    assert vectors == [[1.0, 0.0], [0.0, 1.0]]

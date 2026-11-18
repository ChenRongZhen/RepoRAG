"""LangChain Embeddings 适配器，复用现有 BGE 编码函数。"""

from __future__ import annotations

from langchain_core.embeddings import Embeddings

from app.embedding import encode_documents, encode_query


class BGEEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return encode_documents(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return encode_query(text).tolist()

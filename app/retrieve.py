"""从本地索引中检索与问题最相关的文档片段。"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.build_index import DEFAULT_INDEX_PATH, SCHEMA_VERSION, build_index
from app.embedding import MODEL_NAME, encode_query
from app.load_docs import DEFAULT_SOURCE_DIR, DocumentChunk, content_fingerprint


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float


@lru_cache(maxsize=4)
def load_index(index_path: Path = DEFAULT_INDEX_PATH):
    """加载并校验索引；同一进程同一路径只读一次。"""
    path = Path(index_path)

    if not path.exists():
        print("尚未发现索引，正在首次构建……")
        build_index(index_path=path)

    saved = joblib.load(path)
    if saved.get("model_name") != MODEL_NAME:
        raise ValueError("索引使用的 Embedding 模型与当前配置不一致，请重新构建索引")
    if saved.get("schema_version", 0) > SCHEMA_VERSION:
        raise ValueError("索引版本比当前代码新，请升级代码或重新构建索引")

    saved_fingerprint = saved.get("content_fingerprint")
    if saved_fingerprint and saved_fingerprint != content_fingerprint(DEFAULT_SOURCE_DIR):
        print(
            "提示：CoreCoder 文档已变化，索引可能过期，建议运行 "
            "python -m app.build_index 重建。",
            file=sys.stderr,
        )

    return saved["embeddings"], saved["chunks"]


def clear_index_cache() -> None:
    load_index.cache_clear()


def _rank(query: str, top_k: int, index_path: Path) -> tuple[np.ndarray, list[DocumentChunk], np.ndarray]:
    if not query.strip():
        raise ValueError("问题不能为空")
    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")

    embeddings, chunks = load_index(Path(index_path))
    query_vector = encode_query(query)
    # 文档和问题向量都已经归一化，点积就等于余弦相似度。
    scores = embeddings @ query_vector
    best_indices = scores.argsort()[::-1][: min(top_k, len(chunks))]
    return scores, chunks, best_indices


def retrieve_with_scores(
    query: str,
    top_k: int = 3,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> list[Document]:
    """返回带 source/chunk_id/score 元数据的 LangChain Document。"""
    scores, chunks, best_indices = _rank(query, top_k, index_path)
    documents = []
    for index in best_indices:
        chunk = chunks[index]
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "score": float(scores[index]),
                    "kind": "doc",
                },
            )
        )
    return documents


def search(
    query: str,
    top_k: int = 3,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> list[SearchResult]:
    """返回与 query 最相关的 top_k 个片段（保留旧接口）。"""
    scores, chunks, best_indices = _rank(query, top_k, index_path)
    return [
        SearchResult(chunk=chunks[index], score=float(scores[index]))
        for index in best_indices
    ]


class BgeRetriever(BaseRetriever):
    """LangChain Retriever 适配器，排序与 search() 完全一致。"""

    top_k: int = 3
    index_path: Path = DEFAULT_INDEX_PATH

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return retrieve_with_scores(query, self.top_k, self.index_path)


def print_results(query: str, results: list[SearchResult]) -> None:
    print(f"\n问题：{query}")
    print(f"找到 {len(results)} 个相关片段：")

    for position, result in enumerate(results, start=1):
        preview = result.chunk.text.replace("\n", " ").strip()
        print("\n" + "=" * 72)
        print(f"结果 {position}｜相关度：{result.score:.3f}")
        print(f"来源：{result.chunk.source}｜片段：{result.chunk.chunk_id}")
        print("-" * 72)
        print(preview[:600])


if __name__ == "__main__":
    user_query = input("请输入问题：").strip()
    print_results(user_query, search(user_query))

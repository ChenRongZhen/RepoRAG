"""加载中文 Embedding 模型并生成归一化语义向量。"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "BAAI/bge-small-zh-v1.5"
MODEL_CACHE_DIR = PROJECT_ROOT / "models"
LOCAL_MODEL_DIR = MODEL_CACHE_DIR / "bge-small-zh-v1.5"
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """加载模型；同一个进程中只加载一次。"""
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # 优先读取项目中的完整模型副本，确保后续查询不依赖网络。
    if (LOCAL_MODEL_DIR / "modules.json").exists():
        return SentenceTransformer(str(LOCAL_MODEL_DIR), device="cpu")

    model = SentenceTransformer(
        MODEL_NAME,
        cache_folder=str(MODEL_CACHE_DIR),
        device="cpu",
    )
    model.save(str(LOCAL_MODEL_DIR))
    return model


def encode_documents(texts: list[str]) -> np.ndarray:
    """将文档片段转换成单位长度的稠密向量。"""
    return get_model().encode(
        texts,
        batch_size=16,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


def encode_query(query: str) -> np.ndarray:
    """将短问题转换成向量；BGE建议检索问题添加查询指令。"""
    instructed_query = QUERY_INSTRUCTION + query
    return get_model().encode(
        [instructed_query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]


"""使用中文 Embedding 模型建立本地语义检索索引。"""

from __future__ import annotations

from pathlib import Path

import joblib

from app.embedding import MODEL_NAME, encode_documents
from app.load_docs import DEFAULT_SOURCE_DIR, content_fingerprint, load_documents

SCHEMA_VERSION = 2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = PROJECT_ROOT / "storage" / "embedding_index.joblib"


def build_index(
    source_dir: Path = DEFAULT_SOURCE_DIR,
    index_path: Path = DEFAULT_INDEX_PATH,
) -> Path:
    """读取文档，生成稠密语义向量并原子保存索引。"""
    chunks = load_documents(source_dir)
    texts = [chunk.text for chunk in chunks]
    embeddings = encode_documents(texts)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "content_fingerprint": content_fingerprint(source_dir),
        "embeddings": embeddings,
        "chunks": chunks,
        "source_dir": str(source_dir.resolve()),
    }
    # 先写临时文件再替换：半截写入不会留下坏索引
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    joblib.dump(payload, temporary)
    temporary.replace(index_path)

    print(f"读取文档：{len(set(chunk.source for chunk in chunks))} 个")
    print(f"切分片段：{len(chunks)} 个")
    print(f"Embedding 模型：{MODEL_NAME}")
    print(f"向量矩阵：{embeddings.shape[0]} 行 × {embeddings.shape[1]} 维")
    print(f"索引位置：{index_path}")
    return index_path


if __name__ == "__main__":
    build_index()

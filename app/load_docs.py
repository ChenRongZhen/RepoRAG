"""读取并切分 CoreCoder 的 Markdown 文档。"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)
DEFAULT_SOURCE_DIR = Path(os.getenv("CORECODER_DOCS_DIR") or PROJECT_ROOT.parent / "CoreCoder")


@dataclass
class DocumentChunk:
    """一小段可被检索的文本，以及它来自哪里。"""

    text: str
    source: str
    chunk_id: int


def find_markdown_files(source_dir: Path = DEFAULT_SOURCE_DIR) -> list[Path]:
    """寻找 README_CN.md 和 article 目录下的中文 Markdown 文档。"""
    files: list[Path] = []

    readme = source_dir / "README_CN.md"
    if readme.exists():
        files.append(readme)

    article_dir = source_dir / "article"
    if article_dir.exists():
        files.extend(
            path
            for path in sorted(article_dir.glob("*.md"))
            if not path.stem.endswith("_EN")
        )

    if not files:
        raise FileNotFoundError(
            f"没有在 {source_dir} 找到 README_CN.md 或 article/*.md"
        )
    return files


def split_text(text: str, chunk_size: int = 450, overlap: int = 80) -> list[str]:
    """按字符切分文本，相邻片段保留少量重叠内容。"""
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap 必须大于等于 0 且小于 chunk_size")

    clean_text = text.replace("\r\n", "\n").strip()
    chunks: list[str] = []
    start = 0

    while start < len(clean_text):
        end = min(start + chunk_size, len(clean_text))

        # 尽量在段落或句号附近结束，避免从一句话中间切断。
        if end < len(clean_text):
            candidates = [
                clean_text.rfind("\n\n", start + chunk_size // 2, end),
                clean_text.rfind("。", start + chunk_size // 2, end),
                clean_text.rfind("\n", start + chunk_size // 2, end),
            ]
            natural_end = max(candidates)
            if natural_end > start:
                end = natural_end + 1

        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= len(clean_text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def content_fingerprint(source_dir: Path = DEFAULT_SOURCE_DIR) -> str:
    """语料内容指纹：路径 + 字节。搬家不改指纹，改文件才改。"""
    digest = hashlib.sha256()
    for file_path in find_markdown_files(source_dir):
        digest.update(file_path.relative_to(source_dir).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(file_path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def load_documents(source_dir: Path = DEFAULT_SOURCE_DIR) -> list[DocumentChunk]:
    """读取所有文档，返回带来源信息的文本片段。"""
    chunks: list[DocumentChunk] = []

    for file_path in find_markdown_files(source_dir):
        text = file_path.read_text(encoding="utf-8")
        relative_source = file_path.relative_to(source_dir).as_posix()

        for chunk_id, chunk_text in enumerate(split_text(text)):
            chunks.append(
                DocumentChunk(
                    text=chunk_text,
                    source=relative_source,
                    chunk_id=chunk_id,
                )
            )

    return chunks


if __name__ == "__main__":
    loaded = load_documents()
    print(f"读取并切分完成，共得到 {len(loaded)} 个文本片段。")
    print(f"第一个片段来自：{loaded[0].source}")
    print(loaded[0].text[:300])


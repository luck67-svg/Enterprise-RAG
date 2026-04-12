import hashlib
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.config import settings
from app.rag.loaders import load_file, SUPPORTED_EXTENSIONS
from app.rag.splitter import split_parent_child
from app.rag.vectorstore import get_vectorstore, get_client

# 已上传文件的哈希缓存: {filename: sha256_hex}
_file_hashes: dict[str, str] = {}

router = APIRouter(prefix="/kb", tags=["kb"])


def _compute_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _init_file_hashes() -> None:
    """启动时从已上传文件构建哈希缓存。"""
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        return
    for f in upload_dir.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            _file_hashes[f.name] = _compute_hash(f.read_bytes())


# 初始化缓存
_init_file_hashes()


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "缺少文件名")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"不支持的文件类型: {suffix}，支持: {', '.join(SUPPORTED_EXTENSIONS)}")

    content = await file.read()
    file_hash = _compute_hash(content)

    # 检查是否重复文件（同名同内容）
    if file.filename in _file_hashes and _file_hashes[file.filename] == file_hash:
        logger.info(f"skip duplicate: {file.filename}")
        return {"file": file.filename, "chunks": 0, "ids": 0, "skipped": True, "reason": "文件内容未变化，跳过上传"}

    dest = Path(settings.upload_dir) / file.filename
    dest.write_bytes(content)
    logger.info(f"saved upload: {dest}")

    # 如果同名文件内容变了，先删除旧向量
    if file.filename in _file_hashes:
        _delete_vectors_by_source(file.filename)
        logger.info(f"replaced old vectors: {file.filename}")

    try:
        docs = load_file(dest)
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(422, f"文档解析失败: {e}")

    chunks = split_parent_child(docs)
    for c in chunks:
        c.metadata["source"] = file.filename

    try:
        ids = get_vectorstore().add_documents(chunks)
    except Exception as e:
        err = str(e).lower()
        if "connect" in err and "6333" in err:
            raise HTTPException(503, "Qdrant 不可达，请检查 Docker 容器是否已启动")
        if "ollama" in err or "11434" in err or "embed" in err:
            raise HTTPException(503, "Embedding 调用失败，请检查 Ollama 和 bge-m3 模型是否可用")
        raise HTTPException(503, f"向量入库失败: {e}")

    _file_hashes[file.filename] = file_hash
    return {"file": file.filename, "chunks": len(chunks), "ids": len(ids)}


@router.get("/documents")
def list_documents():
    """列出已上传的文档及其 chunk 数量。"""
    upload_dir = Path(settings.upload_dir)
    if not upload_dir.exists():
        return {"documents": []}

    files = sorted(upload_dir.iterdir())
    docs = []
    for f in files:
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            docs.append({"filename": f.name, "size_kb": round(f.stat().st_size / 1024, 1)})
    return {"documents": docs}


def _delete_vectors_by_source(filename: str) -> None:
    """删除 Qdrant 中指定 source 的所有向量。"""
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    get_client().delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=filename))]
        ),
    )


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    """删除指定文档及其在 Qdrant 中的向量。"""
    filepath = Path(settings.upload_dir) / filename
    if filepath.exists():
        filepath.unlink()

    _delete_vectors_by_source(filename)
    _file_hashes.pop(filename, None)
    logger.info(f"deleted document: {filename}")
    return {"deleted": filename}


@router.get("/stats")
def collection_stats():
    """返回 Qdrant collection 统计信息。"""
    client = get_client()
    try:
        info = client.get_collection(settings.qdrant_collection)
        return {
            "collection": settings.qdrant_collection,
            "vectors_count": info.points_count,
            "points_count": info.points_count,
        }
    except Exception:
        return {"collection": settings.qdrant_collection, "vectors_count": 0, "points_count": 0}

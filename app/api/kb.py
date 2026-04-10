from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.config import settings
from app.rag.loaders import load_file, SUPPORTED_EXTENSIONS
from app.rag.splitter import split_documents
from app.rag.vectorstore import get_vectorstore, get_client

router = APIRouter(prefix="/kb", tags=["kb"])


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "缺少文件名")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(415, f"不支持的文件类型: {suffix}，支持: {', '.join(SUPPORTED_EXTENSIONS)}")

    dest = Path(settings.upload_dir) / file.filename
    dest.write_bytes(await file.read())
    logger.info(f"saved upload: {dest}")

    docs = load_file(dest)
    chunks = split_documents(docs)
    for c in chunks:
        c.metadata["source"] = file.filename

    ids = get_vectorstore().add_documents(chunks)
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


@router.delete("/documents/{filename}")
def delete_document(filename: str):
    """删除指定文档及其在 Qdrant 中的向量。"""
    # 删除文件
    filepath = Path(settings.upload_dir) / filename
    if filepath.exists():
        filepath.unlink()

    # 删除 Qdrant 中对应的向量
    client = get_client()
    from qdrant_client.http.models import Filter, FieldCondition, MatchValue

    client.delete(
        collection_name=settings.qdrant_collection,
        points_selector=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=filename))]
        ),
    )
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

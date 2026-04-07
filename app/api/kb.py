from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from loguru import logger

from app.config import settings
from app.rag.loaders import load_file
from app.rag.splitter import split_documents
from app.rag.vectorstore import get_vectorstore

router = APIRouter(prefix="/kb", tags=["kb"])


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(400, "missing filename")
    dest = Path(settings.upload_dir) / file.filename
    dest.write_bytes(await file.read())
    logger.info(f"saved upload: {dest}")

    try:
        docs = load_file(dest)
    except ValueError as e:
        raise HTTPException(415, str(e))

    chunks = split_documents(docs)
    for c in chunks:
        c.metadata["source"] = file.filename

    ids = get_vectorstore().add_documents(chunks)
    return {"file": file.filename, "chunks": len(chunks), "ids": len(ids)}

from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def load_file(path: str | Path) -> list[Document]:
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return PyMuPDFLoader(str(path)).load()

    if suffix == ".docx":
        return _load_docx(path)

    if suffix == ".txt":
        return _load_txt(path)

    raise ValueError(f"不支持的文件类型: {suffix}，支持: {', '.join(SUPPORTED_EXTENSIONS)}")


def _load_docx(path: Path) -> list[Document]:
    from docx import Document as DocxDocument

    doc = DocxDocument(str(path))
    texts: list[str] = []
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip()
        if text:
            texts.append(text)

    # 按段落合并为单个文档，保留文件名元数据
    full_text = "\n\n".join(texts)
    return [Document(page_content=full_text, metadata={"source": path.name, "page": 1})]


def _load_txt(path: Path) -> list[Document]:
    text = path.read_text(encoding="utf-8")
    return [Document(page_content=text, metadata={"source": path.name, "page": 1})]

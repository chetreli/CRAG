from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
import chardet


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def detect_encoding(file_path: Path) -> str:
    with open(file_path, "rb") as f:
        raw = f.read(10000)
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def load_pdf(file_path: Path) -> list[Document]:
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_type"] = "pdf"
    return docs


def load_text(file_path: Path) -> list[Document]:
    encoding = detect_encoding(file_path)
    loader = TextLoader(str(file_path), encoding=encoding)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = str(file_path)
        doc.metadata["file_type"] = file_path.suffix.lstrip(".")
    return docs


def load_document(file_path: Path) -> list[Document]:
    suffix = file_path.suffix.lower()

    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Неподдерживаемый формат: {suffix}")

    if suffix == ".pdf":
        return load_pdf(file_path)
    else:
        return load_text(file_path)


def load_directory(dir_path: Path) -> list[Document]:
    all_docs = []
    files = [
        f for f in dir_path.rglob("*")
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    
    print(f"Найдено файлов: {len(files)}")
    for file in files:
        try:
            docs = load_document(file)
            all_docs.extend(docs)
            print(f" {file.name} — {len(docs)} страниц/блоков")
        except Exception as e:
            print(f" {file.name} — ошибка: {e}")

    return all_docs
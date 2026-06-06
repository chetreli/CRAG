from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path

RUSSIAN_SEPARATORS = [
    "\n\n",   
    "\n",     
    ". ",     
    "! ",
    "? ",
    "… ",
    ", ",
    " ",
    "",
]


def get_chunker(
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=RUSSIAN_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    chunker = get_chunker(chunk_size, chunk_overlap)
    chunks = chunker.split_documents(documents)

    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "")
        chunk.metadata["chunk_id"] = i
        chunk.metadata["chunk_size"] = len(chunk.page_content)
        chunk.metadata["file_name"] = Path(source).name if source else "unknown"

    print(f"Чанков создано: {len(chunks)} из {len(documents)} документов")
    return chunks
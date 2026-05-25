from dataclasses import dataclass
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_core.documents import Document
from typing import Literal


@dataclass
class ChunkingConfig:
    strategy: Literal["recursive", "semantic", "hybrid"] = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_words: int = 35


def build_chunks(
    doc_unico: list[Document],
    config: ChunkingConfig,
    embeddings=None,
) -> list[Document]:
    if config.strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        chunks = splitter.split_documents(doc_unico)

    elif config.strategy == "semantic":
        assert embeddings is not None, "semantic requiere embeddings"
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        chunks = splitter.split_documents(doc_unico)

    elif config.strategy == "hybrid":
        # Semántico primero, luego recursivo para controlar tamaño máximo
        assert embeddings is not None, "hybrid requiere embeddings"
        semantic = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        recursive = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
        )
        chunks = recursive.split_documents(
            semantic.split_documents(doc_unico)
        )

    # Filtro mínimo de palabras (elimina basura de PDF)
    return [
        c for c in chunks
        if len(c.page_content.split()) >= config.min_words
    ]

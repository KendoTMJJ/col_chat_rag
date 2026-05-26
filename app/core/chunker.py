import re
from dataclasses import dataclass, field
from typing import Literal

from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ─── Configuración ────────────────────────────────────────────────────────────

@dataclass
class ChunkingConfig:
    strategy: Literal["recursive", "semantic", "hybrid"] = "recursive"
    # Mínimo de palabras por chunk (filtro de calidad)
    min_words: int = 35
    target_words: int = 140      # Tamaño ideal de chunk en palabras
    max_words: int = 240         # Tamaño máximo absoluto en palabras
    # Solapamiento en caracteres (usado en recursive/hybrid)
    chunk_overlap: int = 200
    # Oraciones de solapamiento entre chunks (post-procesado)
    overlap_sentences: int = 1


# ─── Entrada pública ──────────────────────────────────────────────────────────

def build_chunks(
    doc_unico: list[Document],
    config: ChunkingConfig | None = None,
    embeddings=None,
) -> list[Document]:
    """
    Genera chunks a partir de una lista de documentos LangChain.

    Estrategias disponibles:
      - recursive : RecursiveCharacterTextSplitter (rápido, sin embeddings)
      - semantic  : SemanticChunker (agrupa por similitud, requiere embeddings)
      - hybrid    : SemanticChunker → RecursiveCharacterTextSplitter (mejor calidad)

    Tras el split inicial se aplican tres mejoras agnósticas al dominio:
      1. Filtro de calidad mínima (min_words).
      2. Overlap por oración completa (overlap_sentences).
      3. Fusión de chunks débiles (muy cortos o con idea incompleta).
    """
    config = config or ChunkingConfig()

    # ── 1. Split inicial según estrategia ────────────────────────────────────
    raw_chunks = _split(doc_unico, config, embeddings)

    # ── 2. Filtro de calidad mínima ──────────────────────────────────────────
    filtered = [c for c in raw_chunks if _count_words(
        c.page_content) >= config.min_words]

    # ── 3. Overlap por oración completa ──────────────────────────────────────
    with_overlap = _apply_sentence_overlap(filtered, config.overlap_sentences)

    # ── 4. Fusión de chunks débiles ──────────────────────────────────────────
    return _merge_weak_chunks(with_overlap, config)


# ─── Split por estrategia ─────────────────────────────────────────────────────

def _split(
    doc_unico: list[Document],
    config: ChunkingConfig,
    embeddings,
) -> list[Document]:
    # Convertimos target_words a caracteres aproximados para LangChain
    # (promedio de 5.5 caracteres por palabra en español)
    approx_chars = int(config.max_words * 5.5)

    if config.strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=approx_chars,
            chunk_overlap=config.chunk_overlap,
        )
        return splitter.split_documents(doc_unico)

    if config.strategy == "semantic":
        assert embeddings is not None, "La estrategia 'semantic' requiere embeddings."
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        return splitter.split_documents(doc_unico)

    if config.strategy == "hybrid":
        assert embeddings is not None, "La estrategia 'hybrid' requiere embeddings."
        semantic = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95,
        )
        recursive = RecursiveCharacterTextSplitter(
            chunk_size=approx_chars,
            chunk_overlap=config.chunk_overlap,
        )
        return recursive.split_documents(semantic.split_documents(doc_unico))

    raise ValueError(f"Estrategia desconocida: '{config.strategy}'")


# ─── Overlap por oración completa ────────────────────────────────────────────

def _apply_sentence_overlap(chunks: list[Document], overlap_sentences: int) -> list[Document]:
    """
    Añade la(s) última(s) oración(es) del chunk anterior al inicio del siguiente.
    Evita el problema del overlap por caracteres que corta oraciones a la mitad.
    """
    if overlap_sentences <= 0 or len(chunks) < 2:
        return chunks

    result: list[Document] = [chunks[0]]

    for i in range(1, len(chunks)):
        prev_sentences = _split_sentences(chunks[i - 1].page_content)
        overlap_text = " ".join(prev_sentences[-overlap_sentences:]).strip()

        if overlap_text:
            new_content = overlap_text + " " + chunks[i].page_content
            result.append(Document(
                page_content=new_content.strip(),
                metadata=chunks[i].metadata,
            ))
        else:
            result.append(chunks[i])

    return result


# ─── Fusión de chunks débiles ─────────────────────────────────────────────────

def _merge_weak_chunks(chunks: list[Document], config: ChunkingConfig) -> list[Document]:
    """
    Fusiona un chunk con el siguiente si:
      - Tiene menos palabras que min_words, o
      - Termina con puntuación que sugiere una idea incompleta (, ; :)
    Solo fusiona si el resultado no supera max_words.
    """
    if not chunks:
        return []

    queue = list(chunks)
    result: list[Document] = []

    i = 0
    while i < len(queue):
        chunk = queue[i]
        words = _count_words(chunk.page_content)
        ends_incomplete = bool(
            re.search(r"[,;:]$", chunk.page_content.strip()))

        if i + 1 < len(queue) and (words < config.min_words or ends_incomplete):
            next_chunk = queue[i + 1]
            combined_words = words + _count_words(next_chunk.page_content)

            if combined_words <= config.max_words:
                merged_content = chunk.page_content.strip() + "\n\n" + \
                    next_chunk.page_content.strip()
                queue[i + 1] = Document(
                    page_content=merged_content,
                    metadata=next_chunk.metadata,
                )
                i += 1
                continue

        result.append(chunk)
        i += 1

    return result


# ─── Utilidades ──────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Separación de oraciones compatible con español."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _count_words(text: str) -> int:
    """Cuenta palabras incluyendo tildes y ñ."""
    return len(re.findall(r"\b[\wáéíóúÁÉÍÓÚñÑüÜ]+\b", text))

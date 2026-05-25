import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from app.core.chunker import ChunkingConfig, build_chunks
from app.core.embedding import generar_embedding, LocalEmbeddings
from app.core.supabase_client import supabase

router = APIRouter()


@router.post("/")
async def subir_pdf(file: UploadFile = File(...)):
    temp_path = None
    try:
        # 1. Archivo temporal
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # 2. Cargar PDF
        docs = PyPDFLoader(temp_path).load()
        texto_completo = "\n".join(d.page_content for d in docs)
        doc_unico = [Document(page_content=texto_completo)]

        # 3. Chunking configurable
        config = ChunkingConfig(strategy="recursive", min_words=35)
        chunks = build_chunks(doc_unico, config)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="El PDF no produjo chunks válidos. Verifica que tenga texto extraíble."
            )

        # 4. Embeddings e inserción
        document_id = str(uuid.uuid4())
        data = [
            {
                "id": str(uuid.uuid4()),
                "texto": chunk.page_content,
                "metadatos": {
                    "document_id": document_id,
                    "filename": file.filename,
                    "page": chunk.metadata.get("page"),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "word_count": len(chunk.page_content.split()),
                },
                "embedding": generar_embedding(chunk.page_content),
            }
            for i, chunk in enumerate(chunks)
        ]

        supabase.table("documentos").insert(data).execute()

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "mensaje": "Documento procesado correctamente",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.core.embedding import generar_embedding
from app.core.supabase_client import supabase

router = APIRouter()


@router.post("/")
async def subir_pdf(file: UploadFile = File(...)):
    try:
        # 1. Guardar archivo temporal
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
            temp.write(await file.read())
            temp_path = temp.name

        # 2. Cargar PDF
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        # 3. Dividir en chunks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(docs)

        document_id = str(uuid.uuid4())

        data = []

        for i, chunk in enumerate(chunks):
            contenido = chunk.page_content
            embedding = generar_embedding(contenido)

            metadata = {
                "document_id": document_id,
                "filename": file.filename,
                "page": chunk.metadata.get("page"),
                "chunk_index": i,
                "total_chunks": len(chunks)
            }

            data.append({
                "id": str(uuid.uuid4()),
                "texto": contenido,
                "metadatos": metadata,
                "embedding": embedding
            })

        supabase.table("documentos").insert(data).execute()

        return {
            "document_id": document_id,
            "filename": file.filename,
            "chunks": len(chunks),
            "mensaje": "Documento procesado correctamente"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from tempfile import NamedTemporaryFile

from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

from app.core.embedding import generar_embedding, LocalEmbeddings
from app.core.supabase_client import supabase
from langchain_core.documents import Document


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

        texto_completo = "\n".join([d.page_content for d in docs])
        doc_unico = [Document(page_content=texto_completo)]

        # 3. Elegir splitter (solo uno de los dos)

        # # OPCIÓN A — Semántico (agrupa frases por similitud, mejor calidad)
        # splitter = SemanticChunker(
        #     embeddings=LocalEmbeddings(),        # <-- objeto Embeddings, no función
        #     breakpoint_threshold_type="percentile",
        #     breakpoint_threshold_amount=95
        # )

        # OPCIÓN B — Recursivo(más rápido, sin llamadas al modelo)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(doc_unico)

        document_id = str(uuid.uuid4())
        data = []

        for i, chunk in enumerate(chunks):
            contenido = chunk.page_content
            # reutiliza el modelo ya cargado
            embedding = generar_embedding(contenido)

            data.append({
                "id": str(uuid.uuid4()),
                "texto": contenido,
                "metadatos": {
                    "document_id": document_id,
                    "filename": file.filename,
                    "page": chunk.metadata.get("page"),
                    "chunk_index": i,
                    "total_chunks": len(chunks)
                },
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

# @router.post("/")
# async def subir_pdf(file: UploadFile = File(...)):
#     try:
#         # 1. Guardar archivo temporal
#         with NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
#             temp.write(await file.read())
#             temp_path = temp.name

#         # 2. Cargar PDF
#         loader = PyPDFLoader(temp_path)
#         docs = loader.load()

#         texto_completo = "\n".join([d.page_content for d in docs])
#         doc_unico = [Document(page_content=texto_completo)]

#         # 3. Paso 1: cortes semánticos (respeta continuidad entre páginas)
#         semantic_splitter = SemanticChunker(
#             embeddings=LocalEmbeddings(),
#             breakpoint_threshold_type="percentile",
#             breakpoint_threshold_amount=95
#         )
#         chunks_semanticos = semantic_splitter.split_documents(doc_unico)

#         # 4. Paso 2: subdivide chunks grandes sin romper semántica
#         recursive_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200
#         )
#         chunks = recursive_splitter.split_documents(chunks_semanticos)

#         document_id = str(uuid.uuid4())
#         data = []

#         for i, chunk in enumerate(chunks):
#             contenido = chunk.page_content
#             embedding = generar_embedding(contenido)

#             data.append({
#                 "id": str(uuid.uuid4()),
#                 "texto": contenido,
#                 "metadatos": {
#                     "document_id": document_id,
#                     "filename": file.filename,
#                     "page": chunk.metadata.get("page"),
#                     "chunk_index": i,
#                     "total_chunks": len(chunks)
#                 },
#                 "embedding": embedding
#             })

#         supabase.table("documentos").insert(data).execute()

#         return {
#             "document_id": document_id,
#             "filename": file.filename,
#             "chunks": len(chunks),
#             "mensaje": "Documento procesado correctamente"
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

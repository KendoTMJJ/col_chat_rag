from app.routes import documents, search, rag, rag_memory
from fastapi import FastAPI

app = FastAPI()

app.include_router(documents.router, prefix="/documentos")
app.include_router(search.router, prefix="/buscar")
app.include_router(rag.router, prefix="/rag")
app.include_router(rag_memory.router, prefix="/rag_memory")

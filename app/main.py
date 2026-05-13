from app.routes import documents, search, rag, rag_re_ranker
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/documentos")
app.include_router(search.router, prefix="/buscar")
app.include_router(rag.router, prefix="/rag")
app.include_router(rag_re_ranker.router, prefix="/rag_memory")

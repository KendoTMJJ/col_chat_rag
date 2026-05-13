from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from typing import List

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')


class LocalEmbeddings(Embeddings):
    """Wrapper compatible con LangChain para SentenceTransformer."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return model.encode(text, show_progress_bar=False).tolist()


def generar_embedding(texto: str) -> List[float]:
    return model.encode(texto, show_progress_bar=False).tolist()

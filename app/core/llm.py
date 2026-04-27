from langchain_ollama import OllamaLLM
from app.core.config import OLLAMA_HOST, OLLAMA_MODEL

_model = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST
)


def generar_respuesta(prompt: str) -> str:
    return _model.invoke(prompt)

from typing import Optional
from typing import Dict

from pydantic import BaseModel, Field


class DocumentoRequest(BaseModel):
    texto: str
    metadatos: Dict = Field(default_factory=dict)


class BusquedaRequest(BaseModel):
    consulta: str
    session_id: str = ''
    top_k: int = 3
    image_base64: Optional[str] = None

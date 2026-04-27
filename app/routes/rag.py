from fastapi import APIRouter, HTTPException
from app.routes.search import buscar_documento
from app.models.schemas import BusquedaRequest

from app.core.llm import generar_respuesta

router = APIRouter()


@router.post("/")
async def responder_con_rag(payload: BusquedaRequest):

    resultados = buscar_documento(payload)

    if not resultados or not resultados.get("resultados"):
        raise HTTPException(
            status_code=404, detail="No se encontró contexto relevante")

    # 🔥 mejor concatenación
    contexto_str = "\n\n".join(
        chunk["texto"] for chunk in resultados["resultados"]
    )

    prompt = f"""
        Basándote únicamente en el siguiente contexto, responde a la pregunta del usuario de forma clara y concisa.

        CONTEXTO:
        {contexto_str}

        PREGUNTA:
        {payload.consulta}

        RESPUESTA:
        """

    respuesta = generar_respuesta(prompt)

    return {"respuesta": respuesta}

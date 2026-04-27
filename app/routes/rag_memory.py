from fastapi import APIRouter, HTTPException

from app.routes.search import buscar_documento
from app.models.schemas import BusquedaRequest
from app.core.config import OLLAMA_HOST, OLLAMA_MODEL

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_ollama import OllamaLLM

router = APIRouter()

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST
)

memorias_de_sesion = {}


def obtener_historial_de_mensajes(session_id: str):
    if session_id not in memorias_de_sesion:
        print(f'Creando nueva memoria para la sesión: {session_id}')
        memorias_de_sesion[session_id] = ChatMessageHistory()
    return memorias_de_sesion[session_id]


reescritor_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("user", "Dada la conversación anterior, genera una consulta de búsqueda autónoma sobre el último tema discutido. No respondas la pregunta, solo reformúlala. para poder contestar la nueva intencion teniendo en cuenta el contexto anterior"),
    ("user", "{input}")
])

cadena_reescritura = reescritor_prompt | llm

prompt_principal = ChatPromptTemplate.from_messages([
    ("system", '''Eres un asistente de IA experto y servicial. Responde basándote en:

    1. El contexto relevante extraído del documento.
    2. El historial de la conversación.

    Responde de forma clara, concisa y amigable. Si la respuesta no está en el contexto o historial, indícalo amablemente.

    Contexto del documento:
    ---
    {contexto}
    ---'''),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

cadena_conversacional = prompt_principal | llm

cadena_con_historial = RunnableWithMessageHistory(
    cadena_conversacional,
    obtener_historial_de_mensajes,
    input_messages_key="input",
    history_messages_key="history",
)


@router.post('/')
async def responder_con_rag_y_memoria(payload: BusquedaRequest):
    try:
        historial = obtener_historial_de_mensajes(payload.session_id)

        if historial.messages:
            consulta_reescrita = await cadena_reescritura.ainvoke({
                "history": historial.messages,
                "input": payload.consulta
            })
            print(
                f"Consulta original: '{payload.consulta}' -> Reescrita: '{consulta_reescrita}'")
        else:
            consulta_reescrita = payload.consulta
            print(f"Primer mensaje, sin reescritura: '{consulta_reescrita}'")

        # Buscar contexto con la consulta reescrita
        payload_busqueda = BusquedaRequest(
            consulta=consulta_reescrita,
            session_id=payload.session_id
        )
        contexto_chunks = buscar_documento(payload_busqueda)
        contexto_str = "\n\n".join(
            chunk['texto'] for chunk in contexto_chunks['resultados']
        )

        # Generar respuesta con historial
        config = {"configurable": {"session_id": payload.session_id}}
        respuesta = await cadena_con_historial.ainvoke({
            "input": payload.consulta,
            "contexto": contexto_str,
        }, config=config)

        return {'respuesta': respuesta}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

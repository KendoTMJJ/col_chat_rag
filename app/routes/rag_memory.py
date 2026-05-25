from fastapi import APIRouter, HTTPException

from app.routes.search import buscar_documento
from app.models.schemas import BusquedaRequest
from app.core.config import OLLAMA_HOST, OLLAMA_MODEL

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from app.core.memory import SupabaseChatMemory

from langchain_ollama import OllamaLLM

router = APIRouter()

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST
)

memorias_de_sesion = {}


def obtener_historial_de_mensajes(session_id: str):
    # if session_id not in memorias_de_sesion:
    #     print(f'Creando nueva memoria para la sesión: {session_id}')
    #     memorias_de_sesion[session_id] = ChatMessageHistory()
    # return memorias_de_sesion[session_id]

    return SupabaseChatMemory(session_id)


reescritor_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("user", "Dada la conversación anterior, genera una consulta de búsqueda autónoma sobre el último tema discutido. No respondas la pregunta, solo reformúlala. para poder contestar la nueva intencion teniendo en cuenta el contexto anterior"),
    ("user", "{input}")
])

cadena_reescritura = reescritor_prompt | llm

prompt_principal = ChatPromptTemplate.from_messages([
    ("system", '''Eres el asistente virtual oficial de Colombia Comparte y Latinoamérica Comparte.

Responde siempre en primera persona institucional, como si fueras parte del equipo.
Habla con naturalidad, calidez y seguridad, como un experto que conoce la organización por dentro.

Reglas estrictas:
- Nunca menciones "el documento", "el contexto", "mi conocimiento" ni ninguna fuente.
- Nunca uses frases como "según...", "basándome en...", "de acuerdo con...".
- Si no tienes información sobre algo, di: "Por el momento no tengo esa información, pero puedes contactarnos en comunicaciones@colombiacomparte.com"
- Responde de forma directa, clara y amigable.

Información disponible:
{contexto}'''),
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


PRONOUNS = {"esto","eso","aquello","él","ella","ellos","ellas","ese","esa","esos","esas","ahí","allí","allá","lo","la","los","las","le","les","me","te","se","nos"}

def _necesita_reescritura(consulta: str, tiene_historial: bool) -> bool:
    if not tiene_historial:
        return False
    palabras = consulta.lower().split()
    if len(palabras) >= 8:
        return False
    primera = palabras[0] if palabras else ""
    return primera in PRONOUNS or consulta.strip().endswith("?") and len(palabras) <= 4

@router.post('/')
async def responder_con_rag_y_memoria(payload: BusquedaRequest):
    try:
        historial = obtener_historial_de_mensajes(payload.session_id)

        if _necesita_reescritura(payload.consulta, bool(historial.messages)):
            consulta_reescrita = await cadena_reescritura.ainvoke({
                "history": historial.messages,
                "input": payload.consulta
            })
            print(f"Consulta reescrita: '{payload.consulta}' -> '{consulta_reescrita}'")
        else:
            consulta_reescrita = payload.consulta
            print(f"Sin reescritura: '{consulta_reescrita}'")

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

from fastapi import APIRouter, HTTPException

from app.routes.search import buscar_documento
from app.models.schemas import BusquedaRequest
from app.core.config import OLLAMA_HOST, OLLAMA_MODEL

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from sentence_transformers.cross_encoder import CrossEncoder

from app.core.memory import SupabaseChatMemory

from langchain_ollama import OllamaLLM

router = APIRouter()

llm = OllamaLLM(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_HOST
)

cross_encoder = CrossEncoder('BAAI/bge-reranker-base')


def obtener_historial_de_mensajes(session_id: str):
    return SupabaseChatMemory(session_id)


reescritor_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="history"),
    ("user", "Dada la conversación anterior, genera una consulta de búsqueda que sea autónoma y pueda ser entendida sin el historial. La consulta debe ser sobre el último tema discutido. No respondas la pregunta, solo reformúlala."),
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
     
- Responde ÚNICAMENTE con información presente en el contexto proporcionado.
- Si la pregunta pide algo que no está textualmente en el contexto (como una "visión" que no existe), 
  NO la inventes ni la inferras. Usa la frase de contacto.
- Nunca completes información que no esté explícita.

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


@router.post('/')
async def responder_con_rag_y_memoria(payload: BusquedaRequest):
    try:
        historial = obtener_historial_de_mensajes(payload.session_id)

        # Reescritura solo si hay historial
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

        payload_busqueda = BusquedaRequest(
            consulta=consulta_reescrita,
            session_id=payload.session_id,
            top_k=10
        )
        contexto_chunks = buscar_documento(payload_busqueda)
        print("--- Contenido de los Chunks Recuperados ---")
        for i, chunk in enumerate(contexto_chunks['resultados']):
            print(f"--- Chunk {i + 1} ---")
            print(chunk['texto'])
            print("\n")

        pares_para_rerank = []
        for chunk in contexto_chunks['resultados']:
            pares_para_rerank.append([consulta_reescrita, chunk['texto']])

        puntajes = cross_encoder.predict(pares_para_rerank)

        for i in range(len(contexto_chunks['resultados'])):
            contexto_chunks['resultados'][i]['relevance_score'] = puntajes[i]

        chunks_reordenados = sorted(
            contexto_chunks['resultados'], key=lambda x: x['relevance_score'], reverse=True)

        contexto_final_chunks = chunks_reordenados[:3]

        print("--- Documentos Re-ordenados ---")
        for chunk in contexto_final_chunks:
            print(f"--- Chunk ---")
            print(f"Puntaje: {chunk['relevance_score']:.4f}")
            print(chunk['texto'])
            print("\n")

        contexto_str = "".join(chunk['texto']
                               for chunk in contexto_final_chunks)

        config = {"configurable": {"session_id": payload.session_id}}

        respuesta_generada = await cadena_con_historial.ainvoke({
            "input": payload.consulta,
            "contexto": contexto_str,
        }, config=config)

        return {'respuesta': respuesta_generada}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Importar modelos Pydantic
from agents.common.schemas import NormalizedQueryKeywords, CartoCiudadQuerySchema, CandidateSchema, IntentInfo, RerankOrderSchema

# Importar estados personalizados
from agents.Agent_intention.states_intention import AgentState, GraphStateOutput, GraphStateInput

# Importar prompts
from agents.Agent_intention.prompt_intetion import KEYWORD_EXTRACTION_PROMPT, INTENT_DETECTION_PROMPT, ENRICHED_QUERY_CONSTRUCTION_PROMPT, RERANKER_PROMPT

# Importar herramienta CartoCiudad
from agents.common.tools import search_cartociudad_tool
from agents.common.llm_config import llm, llm_thinking

# Importar utilidades de deduplicación
from agents.common.utils import deduplicate_candidates, reorder_candidates_by_ids

# --- Funciones de Nodos ---

def keyword_extraction_node(state: GraphStateInput) -> Dict[str, Any]:
    """Extrae palabras clave de la consulta del usuario."""
    print("→ Extrayendo palabras clave")
    user_query = state.user_query
    structured_llm = llm.with_structured_output(NormalizedQueryKeywords)
    response = structured_llm.invoke([
        SystemMessage(content=KEYWORD_EXTRACTION_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}")
    ])
    print(f"  🔑 Palabras clave extraídas: {response.keywords}")
    return {"keywords": response.keywords}

def intent_detection_node(state: GraphStateInput) -> Dict[str, Any]:
    """Detecta la intención del usuario en la consulta."""
    print("→ Detectando intención del usuario")
    user_query = state.user_query
    structured_llm = llm_thinking.with_structured_output(IntentInfo)
    response = structured_llm.invoke([
        SystemMessage(content=INTENT_DETECTION_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}")
    ])
    print(f"  🎯 Intención detectada: {response.intent}")
    return {"intent_info": response}

def enriched_query_construction_node(state: AgentState) -> Dict[str, Any]:
    """Construye una consulta enriquecida usando palabras clave e intención."""
    print("→ Construyendo consulta enriquecida")
    user_query = state["user_query"]
    keywords = state.get("keywords", [])
    intent_info: Optional[IntentInfo] = state.get("intent_info")

    if not intent_info:
        print("  ⚠️ Información de intención faltante, usando valores por defecto")
        intent_str = "desconocida"
        justification_str = "Información de intención no disponible."
    else:
        intent_str = intent_info.intent
        justification_str = intent_info.justification

    structured_llm = llm.with_structured_output(CartoCiudadQuerySchema)
    response = structured_llm.invoke([
        SystemMessage(content=ENRICHED_QUERY_CONSTRUCTION_PROMPT),
        HumanMessage(
            content=(
                f"Consulta original del usuario: {user_query}\n"
                f"Palabras clave extraídas: {keywords}\n"
                f"Intención detectada: {intent_str}\n"
                f"Justificación de la intención: {justification_str}\n\n"
                "Construye los parámetros para la API de CartoCiudad usando esta información."
            )
        )
    ])
    print(f"  📋 Consulta enriquecida: '{response.consulta}' (límite: {response.limite})")
    return {"cartociudad_query_params": response}

def call_cartociudad_api_node(state: AgentState) -> Dict[str, Any]:
    """Realiza la llamada a la API de CartoCiudad."""
    print("→ Llamando a la API de CartoCiudad")
    query_params: Optional[CartoCiudadQuerySchema] = state.get("cartociudad_query_params")

    if not query_params:
        print("  ⚠️ Sin parámetros de consulta, omitiendo llamada a API")
        return {"candidates": []}

    try:
        raw_results = search_cartociudad_tool(
            consulta=query_params.consulta,
            limite=query_params.limite or 10,
            municipio=query_params.municipio,
            provincia=query_params.provincia,
        )
        processed_candidates = [CandidateSchema(**res) for res in raw_results]
        deduped_candidates = deduplicate_candidates(processed_candidates)
        print(f"  ✓ Encontrados {len(deduped_candidates)} candidatos únicos")
        return {"candidates": deduped_candidates}
    except Exception as e:
        print(f"  ❌ Error en API CartoCiudad: {e}")
        return {"candidates": []}

def reranker_intention_node(state: AgentState) -> GraphStateOutput:
    """Reordena los candidatos según su relevancia para la consulta e intención."""
    print("→ Reordenando candidatos por relevancia e intención")
    candidates = state["candidates"]
    query_params = state["cartociudad_query_params"]
    user_query = state["user_query"]

    if not candidates:
        print("  ℹ️ Sin candidatos para reordenar")
        return GraphStateOutput(final_candidates=[], cartociudad_query_params=query_params)

    # Guardar candidatos originales para el reordenamiento
    state["original_candidates"] = candidates.copy()

    candidates_json = [c.model_dump() if hasattr(c, 'model_dump') else dict(c) for c in candidates]
    params_json = query_params.model_dump() if hasattr(query_params, 'model_dump') else dict(query_params)

    structured_llm = llm.with_structured_output(RerankOrderSchema)
    response = structured_llm.invoke([
        SystemMessage(content=RERANKER_PROMPT),
        HumanMessage(
            content=(
                f"Consulta original del usuario: {user_query}\n"
                f"Parámetros utilizados: {params_json}\n"
                f"Lista de candidatos:\n{candidates_json}\n"
                "Devuelve la lista ordenada de IDs en el campo 'ordered_ids'."
            )
        )
    ])

    ordered_ids = response.ordered_ids if hasattr(response, 'ordered_ids') else [c.id for c in candidates]
    reranked_candidates = reorder_candidates_by_ids(ordered_ids, state["original_candidates"])

    print(f"  ✓ Reordenados {len(reranked_candidates)} candidatos")
    return GraphStateOutput(final_candidates=reranked_candidates, cartociudad_query_params=query_params)

# --- Definición del Grafo ---
graph_builder = StateGraph(AgentState, input=GraphStateInput, output=GraphStateOutput)

# Agregar nodos para ramas paralelas
graph_builder.add_node("keyword_extraction", keyword_extraction_node)
graph_builder.add_node("intent_detection", intent_detection_node)

# Agregar nodos para fusión y pasos subsecuentes
graph_builder.add_node("construct_enriched_query", enriched_query_construction_node)
graph_builder.add_node("call_cartociudad", call_cartociudad_api_node)
graph_builder.add_node("reranker_intention", reranker_intention_node)

# Definir aristas para procesamiento paralelo
graph_builder.add_edge(START, "keyword_extraction")
graph_builder.add_edge(START, "intent_detection")

graph_builder.add_edge("keyword_extraction", "construct_enriched_query")
graph_builder.add_edge("intent_detection", "construct_enriched_query")

graph_builder.add_edge("construct_enriched_query", "call_cartociudad")
graph_builder.add_edge("call_cartociudad", "reranker_intention")
graph_builder.add_edge("reranker_intention", END)

app_intention = graph_builder.compile()
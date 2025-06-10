from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Importar modelos Pydantic
from agents.common.schemas import NormalizedQueryKeywords, CartoCiudadQuerySchema, CandidateSchema, RerankSchema, RerankOrderSchema

# Importar estados personalizados
from agents.Agent_base.states_base import GraphStateInput, AgentState, GraphStateOutput

# Importar prompts
from agents.Agent_base.prompt_base import NORMALIZATION_PROMPT, QUERY_CONSTRUCTION_PROMPT, RERANKER_PROMPT

# Importar herramienta CartoCiudad
from agents.common.tools import search_cartociudad_tool
from agents.common.llm_config import llm

# Importar utilidades de deduplicación
from agents.common.utils import deduplicate_candidates, reorder_candidates_by_ids

# --- Funciones de Nodos ---

def extract_normalize_node(state: GraphStateInput) -> Dict[str, Any]:
    """Extrae y normaliza palabras clave de la consulta del usuario."""
    print("→ Normalizando consulta y extrayendo palabras clave")
    user_query = state.user_query
    context_from_meta_evaluator = state.context_from_meta_evaluator

    structured_llm = llm.with_structured_output(NormalizedQueryKeywords)
    response = structured_llm.invoke([
        SystemMessage(content=NORMALIZATION_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}"),
        HumanMessage(content=f"Contexto del meta-evaluador: {context_from_meta_evaluator}") if context_from_meta_evaluator else ""
    ])

    print(f"  Consulta normalizada: {response.normalized_query}")
    print(f"  Palabras clave: {response.keywords}")
    return {
        "normalized_query": response.normalized_query,
        "keywords": response.keywords
    }

def query_construction_node(state: AgentState) -> Dict[str, Any]:
    """Construye los parámetros de consulta para la API de CartoCiudad."""
    print("→ Construyendo parámetros de consulta")
    normalized_query = state.get("normalized_query")
    keywords = state.get("keywords", [])

    if not normalized_query:
        print("  ⚠️ Consulta normalizada faltante, omitiendo construcción")
        return {"cartociudad_query_params": None}

    structured_llm = llm.with_structured_output(CartoCiudadQuerySchema)
    response = structured_llm.invoke([
        SystemMessage(content=QUERY_CONSTRUCTION_PROMPT),
        HumanMessage(
            content=(
                f"Consulta normalizada: {normalized_query}\n"
                f"Palabras clave: {keywords}\n\n"
                "Construye los parámetros para la API de CartoCiudad."
            )
        )
    ])
    
    print(f"  Parámetros generados: consulta='{response.consulta}', límite={response.limite}")
    return {"cartociudad_query_params": response}

def call_cartociudad_api_node(state: AgentState) -> Dict[str, Any]:
    """Realiza la llamada a la API de CartoCiudad y procesa los resultados."""
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
        
        return {
            "candidates": deduped_candidates,
            "cartociudad_query_params": query_params
        }

    except Exception as e:
        print(f"  ❌ Error en API CartoCiudad: {e}")
        return {"candidates": [], "cartociudad_query_params": query_params}

def reranker_base_node(state: AgentState) -> GraphStateOutput:
    """Reordena los candidatos según su relevancia para la consulta."""
    print("→ Reordenando candidatos por relevancia")
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

# Agregar nodos
graph_builder.add_node("extract_normalize", extract_normalize_node)
graph_builder.add_node("construct_query", query_construction_node)
graph_builder.add_node("call_cartociudad", call_cartociudad_api_node)
graph_builder.add_node("reranker_base", reranker_base_node)

# Definir aristas
graph_builder.add_edge(START, "extract_normalize")
graph_builder.add_edge("extract_normalize", "construct_query")
graph_builder.add_edge("construct_query", "call_cartociudad")
graph_builder.add_edge("call_cartociudad", "reranker_base")
graph_builder.add_edge("reranker_base", END)

# Compilar el grafo
app_base = graph_builder.compile()
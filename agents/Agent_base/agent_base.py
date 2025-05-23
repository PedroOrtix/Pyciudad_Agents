from typing import Optional, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# import pydantic models
from agents.common.schemas import NormalizedQueryKeywords, CartoCiudadQuerySchema, CandidateSchema, RerankSchema

# import custom states
from agents.Agent_base.states_base import GraphStateInput, AgentState, GraphStateOutput

# import prompts
from agents.Agent_base.prompt_base import NORMALIZATION_PROMPT, QUERY_CONSTRUCTION_PROMPT, RERANKER_PROMPT

# import cartociudad tool
from agents.common.tools import search_cartociudad_tool
from agents.common.llm_config import llm

# import deduplication utility
from agents.common.utils import deduplicate_candidates

# --- Node Functions ---

def extract_normalize_node(state: GraphStateInput) -> Dict[str, Any]:
    print("--- Running Node: Keyword Extraction & Normalization ---")
    user_query = state.user_query
    context_from_meta_evaluator = state.context_from_meta_evaluator

    structured_llm = llm.with_structured_output(NormalizedQueryKeywords)
    response = structured_llm.invoke([
        SystemMessage(content=NORMALIZATION_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}"),
        HumanMessage(content=f"Contexto del meta-evaluador: {context_from_meta_evaluator}") if context_from_meta_evaluator else ""
    ])

    print(f"Normalized Query: {response.normalized_query}")
    print(f"Keywords: {response.keywords}")
    return {
        "normalized_query": response.normalized_query,
        "keywords": response.keywords
    }

def query_construction_node(state: AgentState) -> Dict[str, Any]:
    print("--- Running Node: Query Construction ---")
    normalized_query = state.get("normalized_query")
    keywords = state.get("keywords", [])

    if not normalized_query:
        print("Warning: Normalized query is missing. Cannot construct CartoCiudad query.")
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
    
    print(f"CartoCiudad Query Params: {response.model_dump_json(indent=2)}")
    return {"cartociudad_query_params": response}

def call_cartociudad_api_node(state: AgentState) -> Dict[str, Any]:
    print("--- Running Node: Call CartoCiudad API ---")
    query_params: Optional[CartoCiudadQuerySchema] = state.get("cartociudad_query_params")

    if not query_params:
        print("Warning: No CartoCiudad query parameters found. Skipping API call.")
        return {"candidates": []}

    try:
        # Use the imported tool
        raw_results = search_cartociudad_tool(
            consulta=query_params.consulta,
            limite=query_params.limite or 10,
            municipio=query_params.municipio,
            provincia=query_params.provincia,
        )
        
        # Convert raw results to CandidateSchema
        processed_candidates = [CandidateSchema(**res) for res in raw_results]
        deduped_candidates = deduplicate_candidates(processed_candidates)
        
        print(f"Found {len(deduped_candidates)} candidates from CartoCiudad (deduplicated).")
        for cand in deduped_candidates:
            print(f"  - ID: {cand.id}, Type: {cand.type}, Address: {cand.address}")
        
        # Instead of returning, pass to next node
        return {
            "candidates": deduped_candidates,
            "cartociudad_query_params": query_params
        }

    except Exception as e:
        print(f"Error calling CartoCiudad API: {e}")
        return {"candidates": [], "cartociudad_query_params": query_params}

def reranker_base_node(state: AgentState) -> GraphStateOutput:
    print("--- Running Node: Reranker Base (Re-ranking candidates) ---")
    candidates = state["candidates"]
    query_params = state["cartociudad_query_params"]
    user_query = state["user_query"]

    if not candidates:
        print("Reranker Base: No candidates to rerank.")
        return GraphStateOutput(final_candidates=[], cartociudad_query_params=query_params)

    # Prepara los datos para el LLM
    candidates_json = [c.model_dump() if hasattr(c, 'model_dump') else dict(c) for c in candidates]
    params_json = query_params.model_dump() if hasattr(query_params, 'model_dump') else dict(query_params)

    # Llama al LLM con el prompt de reranking y espera un output estructurado según RerankSchema
    structured_llm = llm.with_structured_output(RerankSchema)
    response = structured_llm.invoke([
        SystemMessage(content=RERANKER_PROMPT),
        HumanMessage(
            content=(
                f"Consulta original del usuario: {user_query}\n"
                f"Parámetros utilizados: {params_json}\n"
                f"Lista de candidatos:\n{candidates_json}\n"
                "Devuelve la lista reordenada de candidatos en el campo 'rerank_candidates' como una lista de objetos."
            )
        )
    ])

    reranked_candidates = response.rerank_candidates if hasattr(response, 'rerank_candidates') else candidates

    print(f"Reranker Base: Devolviendo {len(reranked_candidates)} candidatos reordenados.")
    return GraphStateOutput(final_candidates=reranked_candidates, cartociudad_query_params=query_params)

# --- Graph Definition ---
graph_builder = StateGraph(AgentState, input=GraphStateInput, output=GraphStateOutput)

# Add nodes
graph_builder.add_node("extract_normalize", extract_normalize_node)
graph_builder.add_node("construct_query", query_construction_node)
graph_builder.add_node("call_cartociudad", call_cartociudad_api_node)
graph_builder.add_node("reranker_base", reranker_base_node)

# Define edges
graph_builder.add_edge(START, "extract_normalize")
graph_builder.add_edge("extract_normalize", "construct_query")
graph_builder.add_edge("construct_query", "call_cartociudad")
graph_builder.add_edge("call_cartociudad", "reranker_base")
graph_builder.add_edge("reranker_base", END)

# Compile the graph
app_base = graph_builder.compile()
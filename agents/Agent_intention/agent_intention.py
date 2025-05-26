from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Import Pydantic models
from agents.common.schemas import NormalizedQueryKeywords, CartoCiudadQuerySchema, CandidateSchema, IntentInfo, RerankSchema, RerankOrderSchema

# import custom states
from agents.Agent_intention.states_intention import AgentState, GraphStateOutput, GraphStateInput

# import prompts
from agents.Agent_intention.prompt_intetion import KEYWORD_EXTRACTION_PROMPT, INTENT_DETECTION_PROMPT, ENRICHED_QUERY_CONSTRUCTION_PROMPT, RERANKER_PROMPT

# import cartocidad tool
from agents.common.tools import search_cartociudad_tool
from agents.common.llm_config import llm, llm_thinking

# import deduplication utility
from agents.common.utils import deduplicate_candidates, reorder_candidates_by_ids

# --- Node Functions ---

def keyword_extraction_node(state: GraphStateInput) -> Dict[str, Any]:
    print("--- Running Node: Keyword Extraction ---")
    user_query = state.user_query
    structured_llm = llm.with_structured_output(NormalizedQueryKeywords)
    response = structured_llm.invoke([
        SystemMessage(content=KEYWORD_EXTRACTION_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}")
    ])
    print(f"Extracted Keywords: {response.keywords}")
    return {"keywords": response.keywords}

def intent_detection_node(state: GraphStateInput) -> Dict[str, Any]:
    print("--- Running Node: Intent Detection ---")
    user_query = state.user_query
    structured_llm = llm_thinking.with_structured_output(IntentInfo)
    response = structured_llm.invoke([
        SystemMessage(content=INTENT_DETECTION_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}")
    ])
    print(f"Justification: {response.justification}")
    return {"intent_info": response}

def enriched_query_construction_node(state: AgentState) -> Dict[str, Any]:
    print("--- Running Node: Enriched Query Construction ---")
    user_query = state["user_query"]
    keywords = state.get("keywords", [])
    intent_info: Optional[IntentInfo] = state.get("intent_info")

    if not intent_info:
        print("Warning: Intent information is missing. Cannot construct enriched query effectively.")
        # Fallback or error handling can be added here
        # For simplicity, we'll proceed, but the query might be suboptimal
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
        processed_candidates = [CandidateSchema(**res) for res in raw_results]
        deduped_candidates = deduplicate_candidates(processed_candidates)
        print(f"Found {len(deduped_candidates)} candidates from CartoCiudad (deduplicated).")
        for cand in deduped_candidates:
            print(f"  - ID: {cand.id}, Type: {cand.type}, Address: {cand.address}")
        return {"candidates": deduped_candidates}
    except Exception as e:
        print(f"Error calling CartoCiudad API: {e}")
        return {"candidates": []}

def reranker_intention_node(state: AgentState) -> GraphStateOutput:
    print("--- Running Node: Reranker Intention (Re-ranking candidates) ---")
    candidates = state["candidates"]
    query_params = state["cartociudad_query_params"]
    user_query = state["user_query"]

    if not candidates:
        print("Reranker Intention: No candidates to rerank.")
        return GraphStateOutput(final_candidates=[], cartociudad_query_params=query_params)

    # Guardar los candidatos originales en el estado efímero
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

    print(f"Reranker Intention: Devolviendo {len(reranked_candidates)} candidatos reordenados.")
    return GraphStateOutput(final_candidates=reranked_candidates, cartociudad_query_params=query_params)

# --- Graph Definition ---
graph_builder = StateGraph(AgentState, input=GraphStateInput, output=GraphStateOutput)

# Add nodes for parallel branches
graph_builder.add_node("keyword_extraction", keyword_extraction_node)
graph_builder.add_node("intent_detection", intent_detection_node)

# Add node for merging and subsequent steps
graph_builder.add_node("construct_enriched_query", enriched_query_construction_node)
graph_builder.add_node("call_cartociudad", call_cartociudad_api_node)
graph_builder.add_node("reranker_intention", reranker_intention_node)

graph_builder.add_edge(START, "keyword_extraction")
graph_builder.add_edge(START, "intent_detection")

graph_builder.add_edge("keyword_extraction", "construct_enriched_query")
graph_builder.add_edge("intent_detection", "construct_enriched_query")

graph_builder.add_edge("construct_enriched_query", "call_cartociudad")
graph_builder.add_edge("call_cartociudad", "reranker_intention")
graph_builder.add_edge("reranker_intention", END)

app_intention = graph_builder.compile()
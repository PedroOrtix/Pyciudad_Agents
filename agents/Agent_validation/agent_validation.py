from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Importar agente base
from agents.Agent_base.agent_base import app_base

# Importar modelos Pydantic
from agents.common.schemas import CartoCiudadQuerySchema, CandidateSchema, ValidationOutput, RerankOrderSchema

# Importar estados personalizados
from agents.Agent_validation.states_validation import GraphStateInput, AgentState, GraphStateOutput
from agents.Agent_base.states_base import GraphStateOutput as AgentBaseGraphStateOutput

# Importar prompts
from agents.Agent_validation.prompt_validation import VALIDATOR_AGENT_REFLEXION_PROMPT, REFORMULATION_AGENT_USING_REFLEXION_PROMPT, RERANKER_PROMPT

# Importar herramienta CartoCiudad
from agents.common.tools import search_cartociudad_tool
from agents.common.llm_config import llm, llm_thinking

# Importar utilidades
from agents.common.utils import deduplicate_candidates, reorder_candidates_by_ids

# --- Funciones de Nodos ---

def invoke_agente_base_node(state: GraphStateInput) -> Dict[str, Any]:
    """Ejecuta el agente base y obtiene los resultados iniciales."""
    print("→ Ejecutando agente base")
    user_query = state.user_query
    
    agente_base_result_state: AgentBaseGraphStateOutput = app_base.invoke({"user_query": user_query})

    initial_params_dict = agente_base_result_state.get("cartociudad_query_params")
    initial_candidates_list = agente_base_result_state.get("final_candidates", [])

    initial_params = None
    if initial_params_dict:
        initial_params = CartoCiudadQuerySchema(**initial_params_dict) if isinstance(initial_params_dict, dict) else initial_params_dict

    processed_initial_candidates = []
    if initial_candidates_list:
        for cand_data in initial_candidates_list:
            processed_initial_candidates.append(CandidateSchema(**(cand_data if isinstance(cand_data, dict) else cand_data.model_dump())))

    print(f"  ✓ Agente base completado: {len(processed_initial_candidates)} candidatos iniciales")
    
    return {
        "current_cartociudad_params": initial_params,
        "candidates_current_iteration": processed_initial_candidates,
        "all_candidates_across_iterations": processed_initial_candidates,
        "reformulation_attempts": 0
    }

def call_cartociudad_api_node(state: AgentState) -> Dict[str, Any]:
    """Realiza llamada iterativa a la API de CartoCiudad con parámetros reformulados."""
    print("→ Llamada iterativa a API CartoCiudad")
    query_params: Optional[CartoCiudadQuerySchema] = state.get("current_cartociudad_params")
    previous_all_candidates = state.get("all_candidates_across_iterations", [])

    if not query_params:
        print("  ⚠️ Sin parámetros para llamada iterativa")
        return {"candidates_current_iteration": []}

    try:
        raw_results = search_cartociudad_tool(
            consulta=query_params.consulta,
            limite=query_params.limite or 10,
            municipio=query_params.municipio,
            provincia=query_params.provincia,
        )
        processed_candidates_this_iteration = [CandidateSchema(**res) for res in raw_results]
        
        # Evitar duplicados basados en direcciones ya vistas
        seen_addresses = {cand.address for cand in previous_all_candidates if cand.address}
        newly_added_candidates = []
        for new_cand in processed_candidates_this_iteration:
            if new_cand.address and new_cand.address not in seen_addresses:
                newly_added_candidates.append(new_cand)
                seen_addresses.add(new_cand.address)
            elif not new_cand.address:
                 newly_added_candidates.append(new_cand)

        updated_all_candidates = previous_all_candidates + newly_added_candidates
        
        print(f"  ✓ Encontrados {len(processed_candidates_this_iteration)} candidatos, {len(newly_added_candidates)} nuevos")
        
        return {
            "candidates_current_iteration": processed_candidates_this_iteration,
            "all_candidates_across_iterations": updated_all_candidates
        }
    except Exception as e:
        print(f"  ❌ Error en API CartoCiudad: {e}")
        return {"candidates_current_iteration": []}

def validator_agent_with_results_node(state: AgentState) -> Dict[str, Any]:
    """Valida los resultados y decide si son suficientes o necesitan reformulación."""
    print("→ Validando resultados")
    user_query = state["user_query"]
    current_params = state["current_cartociudad_params"]
    candidates_this_iteration = state.get("candidates_current_iteration", [])

    candidates_summary_for_llm = [cand.model_dump(include={'address', 'type'}) for cand in candidates_this_iteration[:5]]
    num_candidates = len(candidates_this_iteration)

    structured_llm = llm_thinking.with_structured_output(ValidationOutput)
    response = structured_llm.invoke([
        SystemMessage(content=VALIDATOR_AGENT_REFLEXION_PROMPT),
        HumanMessage(
            content=(
                f"Consulta original del usuario: {user_query}\n"
                f"Parámetros de CartoCiudad usados en esta iteración:\n{current_params.model_dump_json(indent=2) if current_params else 'N/A'}\n"
                f"Número de candidatos encontrados en esta iteración: {num_candidates}\n"
                f"Muestra de candidatos (dirección, tipo) de esta iteración:\n{candidates_summary_for_llm if candidates_summary_for_llm else 'Ninguno'}\n\n"
                "Por favor, primero realiza una 'reflexion_interna' detallada sobre la calidad y adecuación de estos resultados. Luego, basándote en esa reflexión, toma una 'decision_final'."
            )
        )
    ])
    
    print(f"  📊 Decisión de validación: {response.decision_final}")

    update_dict = {"validation_output": response}
    if response.decision_final == "Necesita_Reformulacion":
        update_dict["last_failed_params"] = current_params
        update_dict["last_failed_candidates"] = candidates_this_iteration
        update_dict["last_validation_reflexion"] = response.reflexion_interna
    
    return update_dict

def reformulation_agent_with_results_node(state: AgentState) -> Dict[str, Any]:
    """Reformula los parámetros de consulta basándose en la reflexión del validador."""
    print("→ Reformulando parámetros de consulta")
    user_query = state["user_query"]
    
    failed_params = state.get("last_failed_params")
    failed_candidates = state.get("last_failed_candidates", [])
    validator_reflexion = state.get("last_validation_reflexion", "No hay reflexión específica del validador.")
    
    current_attempts = state.get("reformulation_attempts", 0)

    if not failed_params:
        failed_params_str = "No hay parámetros previos disponibles."
    else:
        failed_params_str = failed_params.model_dump_json(indent=2)

    failed_candidates_summary = [cand.model_dump(include={'address', 'type'}) for cand in failed_candidates[:5]]

    structured_llm = llm.with_structured_output(CartoCiudadQuerySchema)
    response = structured_llm.invoke([
        SystemMessage(content=REFORMULATION_AGENT_USING_REFLEXION_PROMPT),
        HumanMessage(
            content=(
                f"Consulta original del usuario: {user_query}\n"
                f"Este será el intento de reformulación número: {current_attempts + 1}\n\n"
                f"Parámetros del intento anterior (que necesita reformulación):\n{failed_params_str}\n"
                f"Resultados del intento anterior (muestra):\n{failed_candidates_summary if failed_candidates_summary else 'Ninguno'}\n"
                f"Reflexión detallada del agente validador sobre el intento anterior:\n{validator_reflexion}\n\n"
                "Tu tarea es generar un NUEVO conjunto de parámetros para la API de CartoCiudad, tomando en cuenta la reflexión del validador. El objetivo es abordar los problemas o ambigüedades identificados en la reflexión para obtener mejores resultados."
            )
        )
    ])
    
    print(f"  ✏️ Reformulación #{current_attempts + 1} completada")
    
    return {
        "current_cartociudad_params": response,
        "reformulation_attempts": current_attempts + 1
    }

# --- Lógica de Aristas Condicionales ---
MAX_REFORMULATIONS = 2

def decide_after_api_call(state: AgentState) -> str:
    """Decide si llamar al validador o finalizar directamente si ya se alcanzó el límite."""
    attempts_done = state.get("reformulation_attempts", 0)
    
    if attempts_done >= MAX_REFORMULATIONS:
        print(f"  ⚠️ Límite de reformulaciones alcanzado ({MAX_REFORMULATIONS}), finalizando directamente sin validar")
        return "finalize_directly"
    else:
        print(f"  📋 Intento {attempts_done + 1}/{MAX_REFORMULATIONS}, procediendo a validar")
        return "validate"

def decide_to_reformulate_or_end(state: AgentState) -> str:
    """Decide si reformular o finalizar basándose en la validación y número de intentos."""
    print("→ Evaluando si reformular o finalizar")
    val_output = state.get("validation_output")
    attempts_done = state.get("reformulation_attempts", 0)

    if not val_output:
        print("  ⚠️ Sin salida de validación, finalizando por seguridad")
        return "prepare_final_output"

    if val_output.decision_final == "Necesita_Reformulacion":
        print(f"  🔄 Reformulando (intento {attempts_done + 1}/{MAX_REFORMULATIONS})")
        return "reformulate"
    elif val_output.decision_final == "Suficiente":
        print(f"  ✅ Resultados suficientes después de {attempts_done} reformulaciones")
        state["max_reformulations_reached_with_insufficient_results"] = False
        return "prepare_final_output"
    else:
        print(f"  ⚠️ Decisión de validación inesperada: '{val_output.decision_final}', finalizando")
        return "prepare_final_output"

def finalize_output_node(state: AgentState) -> Dict[str,Any]:
    """Finaliza con los candidatos de la iteración actual (validador dijo 'Suficiente')."""
    print("→ Finalizando con candidatos de iteración actual")
    return {
        "final_candidates": state.get("candidates_current_iteration", []),
        "final_params_used_for_last_call": state.get("current_cartociudad_params"),
        "num_reformulations_done": state.get("reformulation_attempts", 0),
        "max_reformulations_hit_insufficient": False
    }

def finalize_directly_node(state: AgentState) -> Dict[str,Any]:
    """Finaliza directamente con todos los candidatos cuando se alcanza el límite sin validar."""
    print("→ Finalizando directamente sin validación (límite alcanzado)")
    
    all_cands = state.get("all_candidates_across_iterations", [])
    unique_final_cands = deduplicate_candidates(all_cands)

    print(f"  📋 Candidatos únicos finales: {len(unique_final_cands)}")

    return {
        "final_candidates": unique_final_cands,
        "final_params_used_for_last_call": state.get("current_cartociudad_params"),
        "num_reformulations_done": state.get("reformulation_attempts", 0),
        "max_reformulations_hit_insufficient": True
    }

def reranker_validation_node(state: AgentState) -> GraphStateOutput:
    """Reordena los candidatos finales según su relevancia."""
    print("→ Reordenando candidatos finales")
    candidates = state["final_candidates_used_for_last_call"] if state.get("max_reformulations_hit_insufficient") else state["candidates_current_iteration"]
    query_params = state["current_cartociudad_params"]
    user_query = state["user_query"]

    if not candidates:
        print("  ℹ️ Sin candidatos para reordenar")
        return GraphStateOutput(final_candidates=[], cartociudad_query_params=query_params)

    # Guardar candidatos originales para el reordenamiento
    state["original_candidates"] = candidates.copy()

    candidates_json = [c.model_dump() if hasattr(c, 'model_dump') else dict(c) for c in candidates]

    structured_llm = llm.with_structured_output(RerankOrderSchema)
    response = structured_llm.invoke([
        SystemMessage(content=RERANKER_PROMPT),
        HumanMessage(
            content=(
                f"Consulta original del usuario: {user_query}\n"
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

graph_builder.add_node("invoke_agente_base", invoke_agente_base_node)
graph_builder.add_node("validator_agent", validator_agent_with_results_node)
graph_builder.add_node("reformulater_agent", reformulation_agent_with_results_node)
graph_builder.add_node("call_api_iterative", call_cartociudad_api_node)
graph_builder.add_node("reranker_validation", reranker_validation_node)
graph_builder.add_node("finalize_output", finalize_output_node)
graph_builder.add_node("finalize_directly", finalize_directly_node)

# Definir aristas
graph_builder.add_edge(START, "invoke_agente_base")
graph_builder.add_edge("invoke_agente_base", "validator_agent")

graph_builder.add_conditional_edges(
    "validator_agent",
    decide_to_reformulate_or_end,
    {
        "reformulate": "reformulater_agent",
        "prepare_final_output": "finalize_output"
    }
)

# El reformulador genera nuevos parámetros, luego llamamos a la API y decidimos si validar o finalizar
graph_builder.add_edge("reformulater_agent", "call_api_iterative")

graph_builder.add_conditional_edges(
    "call_api_iterative",
    decide_after_api_call,
    {
        "validate": "validator_agent",
        "finalize_directly": "finalize_directly"
    }
)

# Conexiones finales al reranker
graph_builder.add_edge("finalize_output", "reranker_validation")
graph_builder.add_edge("finalize_directly", "reranker_validation")

graph_builder.add_edge("reranker_validation", END)

app_validation = graph_builder.compile()
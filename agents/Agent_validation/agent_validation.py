from typing import Dict, Any, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# importamos todo lo relacionado con el agente_base
from agents.Agent_base.agent_base import app_base

# Import Pydantic models
from agents.common.schemas import CartoCiudadQuerySchema, CandidateSchema, ValidationOutput

# Import custom states
from agents.Agent_validation.states_validation import GraphStateInput, AgentState, GraphStateOutput
from agents.Agent_base.states_base import GraphStateOutput as AgentBaseGraphStateOutput

# Import prompts
from agents.Agent_validation.prompt_validation import VALIDATOR_AGENT_REFLEXION_PROMPT, REFORMULATION_AGENT_USING_REFLEXION_PROMPT

# Import CartoCiudad tool
from agents.common.tools import search_cartociudad_tool
from agents.common.llm_config import llm

# Import utility function
from agents.common.utils import deduplicate_candidates

# --- Node Functions ---

def invoke_agente_base_node(state: GraphStateInput) -> Dict[str, Any]:
    print("--- [Agente_2] Running Node: Invoke Agente_Base ---")
    user_query = state.user_query
    
    print(f"  Invoking Agente_Base con query: {user_query}")
    # La salida de agente_base.app.invoke() será una instancia de su AgentState (AgentStateBase)
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


    print(f"  Agente_Base Params: {initial_params.model_dump_json(indent=2) if initial_params else 'None'}")
    print(f"  Agente_Base Found {len(processed_initial_candidates)} candidates.")
    
    # Estos son los resultados de la "iteración 0"
    return {
        "current_cartociudad_params": initial_params,
        "candidates_current_iteration": processed_initial_candidates,
        "all_candidates_across_iterations": processed_initial_candidates, # Primera carga
        "reformulation_attempts": 0
    }

def call_cartociudad_api_node(state: AgentState) -> Dict[str, Any]:
    print("--- [Agente_2] Running Node: Call CartoCiudad API (Iterative) ---")
    query_params: Optional[CartoCiudadQuerySchema] = state.get("current_cartociudad_params")
    previous_all_candidates = state.get("all_candidates_across_iterations", [])

    if not query_params:
        print("  Warning: No CartoCiudad query parameters for iterative call. Skipping.")
        return {"candidates_current_iteration": []}

    print(f"  Calling API with params: {query_params.model_dump_json(indent=2)}")
    try:
        # Use the imported tool by unpacking parameters from query_params
        raw_results = search_cartociudad_tool(
            consulta=query_params.consulta,
            limite=query_params.limite or 10,
            municipio=query_params.municipio,
            provincia=query_params.provincia,
        )
        processed_candidates_this_iteration = [CandidateSchema(**res) for res in raw_results]
        print(f"  Found {len(processed_candidates_this_iteration)} candidates in this iteration.")
        
        seen_addresses = {cand.address for cand in previous_all_candidates if cand.address}
        newly_added_candidates = []
        for new_cand in processed_candidates_this_iteration:
            if new_cand.address and new_cand.address not in seen_addresses:
                newly_added_candidates.append(new_cand)
                seen_addresses.add(new_cand.address)
            elif not new_cand.address:
                 newly_added_candidates.append(new_cand)

        updated_all_candidates = previous_all_candidates + newly_added_candidates
        
        return {
            "candidates_current_iteration": processed_candidates_this_iteration,
            "all_candidates_across_iterations": updated_all_candidates
        }
    except Exception as e:
        print(f"  Error calling CartoCiudad API: {e}")
        return {"candidates_current_iteration": []}

def validator_agent_with_results_node(state: AgentState) -> Dict[str, Any]:
    print("--- [Agente_2] Running Node: Validator Agent (with Reflection & Results) ---")
    user_query = state["user_query"]
    current_params = state["current_cartociudad_params"]
    # Usamos los candidatos de la iteración actual para la validación
    candidates_this_iteration = state.get("candidates_current_iteration", [])

    candidates_summary_for_llm = [cand.model_dump(include={'address', 'type'}) for cand in candidates_this_iteration[:5]]
    num_candidates = len(candidates_this_iteration)

    structured_llm = llm.with_structured_output(ValidationOutput)
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
    print(f"  Validator Output: Decision: {response.decision_final}")
    print(f"  Validator Reflexion:\n{response.reflexion_interna}")

    update_dict = {"validation_output": response}
    if response.decision_final == "Necesita_Reformulacion":
        update_dict["last_failed_params"] = current_params
        update_dict["last_failed_candidates"] = candidates_this_iteration # Los que causaron la necesidad de reformular
        update_dict["last_validation_reflexion"] = response.reflexion_interna
    
    return update_dict


def reformulation_agent_with_results_node(state: AgentState) -> Dict[str, Any]:
    print("--- [Agente_2] Running Node: Reformulation Agent (using Validator's Reflection) ---")
    user_query = state["user_query"]
    
    failed_params = state.get("last_failed_params")
    failed_candidates = state.get("last_failed_candidates", [])
    validator_reflexion = state.get("last_validation_reflexion", "No specific validator reflexion provided.")
    
    current_attempts = state.get("reformulation_attempts", 0) # Este es el número de reformulaciones YA HECHAS

    if not failed_params: # Si por alguna razón no hay failed_params (ej. primer validador falló sin params)
        failed_params_str = "No previous parameters were available or they were None."
    else:
        failed_params_str = failed_params.model_dump_json(indent=2)

    failed_candidates_summary = [cand.model_dump(include={'address', 'type'}) for cand in failed_candidates[:5]]

    structured_llm = llm.with_structured_output(CartoCiudadQuerySchema) # Schema de Agente_2
    response = structured_llm.invoke([
        SystemMessage(content=REFORMULATION_AGENT_USING_REFLEXION_PROMPT),
        HumanMessage(
            content=(
                f"Consulta original del usuario: {user_query}\n"
                # El intento que SE VA A HACER es current_attempts + 1
                f"Este será el intento de reformulación número: {current_attempts + 1}\n\n"
                f"Parámetros del intento anterior (que necesita reformulación):\n{failed_params_str}\n"
                f"Resultados del intento anterior (muestra):\n{failed_candidates_summary if failed_candidates_summary else 'Ninguno'}\n"
                f"Reflexión detallada del agente validador sobre el intento anterior:\n{validator_reflexion}\n\n"
                "Tu tarea es generar un NUEVO conjunto de parámetros para la API de CartoCiudad, tomando en cuenta la reflexión del validador. El objetivo es abordar los problemas o ambigüedades identificados en la reflexión para obtener mejores resultados."
            )
        )
    ])
    print(f"  Reformulated CartoCiudad Params: {response.model_dump_json(indent=2)}")
    
    return {
        "current_cartociudad_params": response, # Nuevos parámetros para la siguiente llamada a API
        "reformulation_attempts": current_attempts + 1 # Incrementar el contador de reformulaciones hechas
    }

# --- Conditional Edge Logic ---
MAX_REFORMULATIONS = 2 # 1 intento inicial (de agente_base) + 2 reformulaciones = 3 total API calls

def decide_to_reformulate_or_end(state: AgentState) -> str:
    print("--- [Agente_2] Decision: Reformulate or End? ---")
    val_output = state.get("validation_output")
    # 'reformulation_attempts' cuenta cuántas reformulaciones se han *completado*.
    # Si es 0, significa que estamos evaluando el resultado del agente_base.
    # Si es 1, estamos evaluando el resultado de la 1ª reformulación.
    # Si es 2, estamos evaluando el resultado de la 2ª reformulación.
    attempts_done = state.get("reformulation_attempts", 0)

    if not val_output:
        print("  Warning: No validation output. Ending as fallback.")
        # Si no hay output de validación, es un error, terminamos.
        return "prepare_final_output" # o END directamente

    if val_output.decision_final == "Necesita_Reformulacion":
        if attempts_done < MAX_REFORMULATIONS:
            print(f"  Decision: Needs reformulation (Upcoming attempt {attempts_done + 1}). Routing to reformulation_agent.")
            return "reformulate" # Ir a reformulater_agent
        else:
            # Se alcanzó el máximo de reformulaciones y la última aún no es "Suficiente"
            print(f"  Decision: Max reformulation attempts ({MAX_REFORMULATIONS}) reached and results still insufficient. Preparing final output with all candidates.")
            # Marcamos que terminamos por esta razón
            state["max_reformulations_reached_with_insufficient_results"] = True
            return "prepare_final_output_all_candidates" # Nuevo nodo/ruta
            
    elif val_output.decision_final == "Suficiente":
        print(f"  Decision: Results are sufficient after {attempts_done} reformulations. Preparing final output.")
        state["max_reformulations_reached_with_insufficient_results"] = False
        return "prepare_final_output" # Ir a preparar salida con los candidatos actuales
    else:
        print(f"  Warning: Unexpected validation decision '{val_output.decision_final}'. Preparing final output.")
        return "prepare_final_output"


def finalize_output_node(state: AgentState) -> Dict[str,Any] : # Este es el GraphStateOutput
    print("--- [Agente_2] Running Node: Finalize Output (Current Iteration Candidates) ---")
    # Esta ruta se toma si el validador dijo "Suficiente"
    final_output_payload = {
        "final_candidates": state.get("candidates_current_iteration", []),
        "final_params_used_for_last_call": state.get("current_cartociudad_params"),
        "num_reformulations_done": state.get("reformulation_attempts", 0),
        "max_reformulations_hit_insufficient": False
    }
    return GraphStateOutput(final_candidates=final_output_payload["final_candidates"], final_cartociudad_params=final_output_payload["final_params_used_for_last_call"])


def finalize_output_all_candidates_node(state: AgentState) -> Dict[str,Any] :
    print("--- [Agente_2] Running Node: Finalize Output (All Accumulated Candidates) ---")
    # Esta ruta se toma si se agotaron los reintentos y los resultados no fueron "Suficientes"
    
    # Deduplicación final de todos los candidatos acumulados usando la función utilitaria
    all_cands = state.get("all_candidates_across_iterations", [])
    unique_final_cands = deduplicate_candidates(all_cands)

    final_output_payload = {
        "final_candidates": unique_final_cands,
        "final_params_used_for_last_call": state.get("current_cartociudad_params"), # Params de la última llamada
        "num_reformulations_done": state.get("reformulation_attempts", 0),
        "max_reformulations_hit_insufficient": True
    }
    return GraphStateOutput(final_candidates=final_output_payload["final_candidates"], final_cartociudad_params=final_output_payload["final_params_used_for_last_call"])

# --- Graph Definition ---
graph_builder = StateGraph(AgentState, input=GraphStateInput, output=GraphStateOutput)

graph_builder.add_node("invoke_agente_base", invoke_agente_base_node)

graph_builder.add_node("validator_agent", validator_agent_with_results_node)
graph_builder.add_node("reformulater_agent", reformulation_agent_with_results_node)
graph_builder.add_node("call_api_iterative", call_cartociudad_api_node) # Para las reformulaciones

# Nodos finales
graph_builder.add_node("finalize_output", finalize_output_node)
graph_builder.add_node("finalize_output_all", finalize_output_all_candidates_node)


# Define edges
graph_builder.add_edge(START, "invoke_agente_base")
# Después de invocar agente_base, que ya llamó a su API y obtuvo candidatos,
# vamos directamente a validar esos resultados.
graph_builder.add_edge("invoke_agente_base", "validator_agent")

graph_builder.add_conditional_edges(
    "validator_agent",
    decide_to_reformulate_or_end,
    {
        "reformulate": "reformulater_agent",
        "prepare_final_output": "finalize_output", # Si fue "Suficiente"
        "prepare_final_output_all_candidates": "finalize_output_all" # Si max_attempts y no suficiente
    }
)
# El reformulador genera nuevos params, luego llamamos a la API con ellos, luego validamos de nuevo
graph_builder.add_edge("reformulater_agent", "call_api_iterative")
graph_builder.add_edge("call_api_iterative", "validator_agent")

# Conexiones a END
graph_builder.add_edge("finalize_output", END)
graph_builder.add_edge("finalize_output_all", END)

# Especificar el schema de salida del grafo
app_validation = graph_builder.compile()
from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Importar modelos Pydantic
from agents.common.schemas import PipelineDecisionSchema

# Importar estados del meta-agente
from agents.Agent_ensemble.states_ensemble import MainAgentState, GraphStateOutput, GraphStateInput
from agents.Agent_base.states_base import GraphStateInput as GraphStateInputBase
from agents.Agent_intention.states_intention import GraphStateInput as GraphStateInputIntention
from agents.Agent_validation.states_validation import GraphStateInput as GraphStateInputValidation

# Importar prompts
from agents.Agent_ensemble.prompt_ensemble import META_AGENT_EVALUATOR_PROMPT

# Importar LLMs
from agents.common.llm_config import llm_thinking

# Importar aplicaciones compiladas de subgrafos
from agents.Agent_base.agent_base import app_base
from agents.Agent_intention.agent_intention import app_intention
from agents.Agent_validation.agent_validation import app_validation

# --- Grafo Meta-Agente (Grafo Principal) ---

def meta_agent_evaluator_node(state: GraphStateInput) -> Dict[str, Any]:
    """Evalúa la consulta del usuario y selecciona el pipeline más apropiado."""
    print("🧠 Meta-Agente: Evaluando consulta y seleccionando pipeline")
    user_query = state.user_query
    structured_llm = llm_thinking.with_structured_output(PipelineDecisionSchema)
    response = structured_llm.invoke([
        SystemMessage(content=META_AGENT_EVALUATOR_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}")
    ])
    print(f"  📍 Pipeline seleccionado: {response.selected_pipeline}")
    return {
        "selected_pipeline_name": response.selected_pipeline,
        "pipeline_justification": response.justification
    }

def invoke_pipeline_simple_node(state: GraphStateInputBase) -> Dict[str, Any]:
    """Ejecuta el pipeline simple (agente base)."""
    print("🔧 Ejecutando PIPELINE_SIMPLE")
    subgraph_result = app_base.invoke({"user_query": state.user_query})
    candidates = subgraph_result.get("final_candidates", [])
    print(f"  ✓ Pipeline simple completado: {len(candidates)} candidatos")
    return {
        "final_cartociudad_params": subgraph_result.get("cartociudad_query_params", None),
        "candidates": candidates
    }

def invoke_pipeline_intermedio_node(state: GraphStateInputIntention) -> Dict[str, Any]:
    """Ejecuta el pipeline intermedio (agente de intención)."""
    print("🔧 Ejecutando PIPELINE_INTERMEDIO")
    subgraph_result = app_intention.invoke({"user_query": state.user_query})
    candidates = subgraph_result.get("final_candidates", [])
    print(f"  ✓ Pipeline intermedio completado: {len(candidates)} candidatos")
    return {
        "final_cartociudad_params": subgraph_result.get("cartociudad_query_params", None),
        "candidates": candidates
    }

def invoke_pipeline_complejo_node(state: GraphStateInputValidation) -> Dict[str, Any]:
    """Ejecuta el pipeline complejo (agente de validación)."""
    print("🔧 Ejecutando PIPELINE_COMPLEJO")
    subgraph_result = app_validation.invoke({"user_query": state.user_query})
    candidates = subgraph_result.get("final_candidates", [])
    print(f"  ✓ Pipeline complejo completado: {len(candidates)} candidatos")
    return {
        "final_cartociudad_params": subgraph_result.get("final_cartociudad_params", None),
        "candidates": candidates
    }
    
def validate_decition_node(state: MainAgentState) -> Dict[str, Any]:
    """Valida los resultados del pipeline y decide si escalar o finalizar."""
    print("🔍 Validando resultados del pipeline")

    pipeline = state.get("selected_pipeline_name")
    candidates = state.get("candidates", [])
    justification = state.get("pipeline_justification")
    escalated_from = state.get("escalated_from", [])

    # Si hay resultados, aceptar y finalizar
    if candidates:
        print(f"  ✅ Resultados satisfactorios en {pipeline}: {len(candidates)} candidatos")
        return {
            **state,
            "validation_decision": "aceptar"
        }

    # Si ya hemos pasado por este pipeline antes o estamos en el más complejo
    if pipeline in escalated_from or pipeline == "PIPELINE_COMPLEJO":
        print(f"  ⚠️ Sin más opciones de escalado. Finalizando en {pipeline}")
        return {
            **state,
            "validation_decision": "forzar_fin"
        }

    # Escalar al siguiente pipeline
    next_pipeline = {
        "PIPELINE_SIMPLE": "PIPELINE_INTERMEDIO",
        "PIPELINE_INTERMEDIO": "PIPELINE_COMPLEJO"
    }.get(pipeline, "PIPELINE_COMPLEJO")

    print(f"  ⬆️ Escalando de {pipeline} a {next_pipeline}")
    return {
        **state,
        "selected_pipeline_name": next_pipeline,
        "pipeline_justification": justification + f" | Escalado automático desde {pipeline}",
        "validation_decision": "escalar",
        "escalated_from": escalated_from + [pipeline],
    }

def validate_router(state: MainAgentState) -> str:
    """Enruta según la decisión de validación."""
    decision = state.get("validation_decision")
    pipeline = state.get("selected_pipeline_name")

    if decision == "aceptar" or decision == "forzar_fin":
        return "output_node"
    
    if decision == "escalar":
        if pipeline == "PIPELINE_SIMPLE":
            return "invoke_simple"
        elif pipeline == "PIPELINE_INTERMEDIO":
            return "invoke_intermedio"
        elif pipeline == "PIPELINE_COMPLEJO":
            return "invoke_complejo"

    print(f"  ⚠️ Decisión de routing desconocida: '{decision}'. Finalizando")
    return "output_node"

def output_node(state: MainAgentState) -> GraphStateOutput:
    """Nodo final que retorna los resultados del grafo principal."""
    print("📤 Preparando salida final")
    final_candidates = state.get("candidates", [])
    print(f"  📋 Resultados finales: {len(final_candidates)} candidatos")
    return GraphStateOutput(
        final_candidates=final_candidates,
        cartociudad_query_params=state.get("final_cartociudad_params")
    )

def main_select_pipeline_router(state: MainAgentState) -> str:
    """Router condicional para la selección de pipeline."""
    pipeline_name = state.get("selected_pipeline_name")
    if pipeline_name == "PIPELINE_SIMPLE":
        return "invoke_simple"
    elif pipeline_name == "PIPELINE_INTERMEDIO":
        return "invoke_intermedio"
    elif pipeline_name == "PIPELINE_COMPLEJO":
        return "invoke_complejo"
    print(f"  ⚠️ Pipeline desconocido '{pipeline_name}'. Usando simple por defecto")
    return "invoke_simple"

# --- Construcción del Grafo Principal ---
graph_builder = StateGraph(MainAgentState, input=GraphStateInput, output=GraphStateOutput)

graph_builder.add_node("meta_evaluator", meta_agent_evaluator_node)
graph_builder.add_node("invoke_simple", invoke_pipeline_simple_node)
graph_builder.add_node("invoke_intermedio", invoke_pipeline_intermedio_node)
graph_builder.add_node("invoke_complejo", invoke_pipeline_complejo_node)
graph_builder.add_node("validate_decision", validate_decition_node)
graph_builder.add_node("output_node", output_node)

# Definir flujo: cada pipeline va a validación antes de finalizar
graph_builder.add_edge(START, "meta_evaluator")

graph_builder.add_conditional_edges(
    "meta_evaluator",
    main_select_pipeline_router,
    {
        "invoke_simple": "invoke_simple",
        "invoke_intermedio": "invoke_intermedio",
        "invoke_complejo": "invoke_complejo"
    }
)

graph_builder.add_edge("invoke_simple", "validate_decision")
graph_builder.add_edge("invoke_intermedio", "validate_decision")
graph_builder.add_edge("invoke_complejo", "validate_decision")

graph_builder.add_conditional_edges(
    "validate_decision",
    validate_router,
    {
        "invoke_simple": "invoke_simple",
        "invoke_intermedio": "invoke_intermedio",
        "invoke_complejo": "invoke_complejo",
        "output_node": "output_node"
    }
)

graph_builder.add_edge("output_node", END)

app_ensemble = graph_builder.compile()
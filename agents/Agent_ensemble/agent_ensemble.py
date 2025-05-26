from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Import Pydantic models
from agents.common.schemas import PipelineDecisionSchema

# Import main state for the meta-agent
from agents.Agent_ensemble.states_ensemble import MainAgentState, GraphStateOutput, GraphStateInput
from agents.Agent_base.states_base import GraphStateInput as GraphStateInputBase
from agents.Agent_intention.states_intention import GraphStateInput as GraphStateInputIntention
from agents.Agent_validation.states_validation import GraphStateInput as GraphStateInputValidation
# Import prompts
from agents.Agent_ensemble.prompt_ensemble import META_AGENT_EVALUATOR_PROMPT

# import llms
from agents.common.llm_config import llm_thinking

# Import the compiled subgraph apps from each agent
from agents.Agent_base.agent_base import app_base
from agents.Agent_intention.agent_intention import app_intention
from agents.Agent_validation.agent_validation import app_validation

# --- Meta-Agent Graph (Main Graph) ---
def meta_agent_evaluator_node(state: GraphStateInput) -> Dict[str, Any]:
    print("[MainGraph] > Meta-Agent Evaluator")
    user_query = state.user_query
    structured_llm = llm_thinking.with_structured_output(PipelineDecisionSchema)
    response = structured_llm.invoke([
        SystemMessage(content=META_AGENT_EVALUATOR_PROMPT),
        HumanMessage(content=f"Consulta del usuario: {user_query}")
    ])
    print(f"[MainGraph] > Pipeline Decision: {response.selected_pipeline}")
    return {
        "selected_pipeline_name": response.selected_pipeline,
        "pipeline_justification": response.justification
    }

# Nodes to invoke subgraphs (devuelven resultados de cada pipeline)
def invoke_pipeline_simple_node(state: GraphStateInputBase) -> Dict[str, Any]:
    print("[MainGraph] > Invoking PIPELINE_SIMPLE (app_base)")
    subgraph_result = app_base.invoke({"user_query": state.user_query})
    return {
        "final_cartociudad_params": subgraph_result.get("cartociudad_query_params", None),
        "candidates": subgraph_result.get("final_candidates", [])
    }

def invoke_pipeline_intermedio_node(state: GraphStateInputIntention) -> Dict[str, Any]:
    print("[MainGraph] > Invoking PIPELINE_INTERMEDIO (app_intention)")
    subgraph_result = app_intention.invoke({"user_query": state.user_query})
    return {
        "final_cartociudad_params": subgraph_result.get("cartociudad_query_params", None),
        "candidates": subgraph_result.get("final_candidates", [])
    }

def invoke_pipeline_complejo_node(state: GraphStateInputValidation) -> Dict[str, Any]:
    print("[MainGraph] > Invoking PIPELINE_COMPLEJO (app_validation)")
    subgraph_result = app_validation.invoke({"user_query": state.user_query})
    return {
        "final_cartociudad_params": subgraph_result.get("final_cartociudad_params", None),
        "candidates": subgraph_result.get("final_candidates", [])
    }
    
def validate_decition_node(state: MainAgentState) -> Dict[str, Any]:
    print("[MainGraph] > Validating pipeline decision and results")

    pipeline = state.get("selected_pipeline_name")
    candidates = state.get("candidates", [])
    justification = state.get("pipeline_justification")
    escalated_from = state.get("escalated_from", [])  # Pistas de escalado previo

    # Si hay resultados, aceptar y finalizar
    if candidates:
        print(f"[MainGraph] > Validation: Found {len(candidates)} candidates with pipeline {pipeline}.")
        return {
            **state,
            "validation_decision": "aceptar"
        }

    # Si ya hemos pasado por este pipeline antes, no repetir
    if pipeline in escalated_from or pipeline == "PIPELINE_COMPLEJO":
        print(f"[MainGraph] > Validation: No more escalation possible. Finishing at {pipeline}.")
        return {
            **state,
            "validation_decision": "forzar_fin"
        }

    # Escalamos al siguiente pipeline (si es posible)
    next_pipeline = {
        "PIPELINE_SIMPLE": "PIPELINE_INTERMEDIO",
        "PIPELINE_INTERMEDIO": "PIPELINE_COMPLEJO"
    }.get(pipeline, "PIPELINE_COMPLEJO")

    print(f"[MainGraph] > Validation: Escalating from {pipeline} to {next_pipeline}.")
    return {
        **state,
        "selected_pipeline_name": next_pipeline,
        "pipeline_justification": justification + f" | Escalado automático desde {pipeline}",
        "validation_decision": "escalar",
        "escalated_from": escalated_from + [pipeline],
    }

        
def validate_router(state: MainAgentState) -> str:
    decision = state.get("validation_decision")
    pipeline = state.get("selected_pipeline_name")

    if decision == "aceptar" or decision == "forzar_fin":
        print(f"[MainGraph] > Routing to output_node. Decision: {decision}")
        return "output_node"
    
    if decision == "escalar":
        print(f"[MainGraph] > Routing to next pipeline: {pipeline}")
        if pipeline == "PIPELINE_SIMPLE":
            return "invoke_simple"
        elif pipeline == "PIPELINE_INTERMEDIO":
            return "invoke_intermedio"
        elif pipeline == "PIPELINE_COMPLEJO":
            return "invoke_complejo"

    print(f"[MainGraph] > Unknown routing decision '{decision}'. Ending.")
    return "output_node"  # Fallback to output node

def output_node(state: MainAgentState) -> GraphStateOutput:
    """Final output node to return the results of the main graph."""
    print("[MainGraph] > Output Node")
    return GraphStateOutput(
        final_candidates=state.get("candidates", []),
        cartociudad_query_params=state.get("final_cartociudad_params")
    )

# Conditional Router for Pipeline Selection (in Main Graph)
def main_select_pipeline_router(state: MainAgentState) -> str:
    pipeline_name = state.get("selected_pipeline_name")
    print(f"[MainGraph] > Routing to: {pipeline_name}")
    if pipeline_name == "PIPELINE_SIMPLE":
        return "invoke_simple"
    elif pipeline_name == "PIPELINE_INTERMEDIO":
        return "invoke_intermedio"
    elif pipeline_name == "PIPELINE_COMPLEJO":
        return "invoke_complejo"
    print(f"[MainGraph] > Warning: Unknown pipeline '{pipeline_name}'. Defaulting to simple.")
    return "invoke_simple" # Fallback

# --- Build the Main Graph ---
graph_builder = StateGraph(MainAgentState, input=GraphStateInput, output=GraphStateOutput)

graph_builder.add_node("meta_evaluator", meta_agent_evaluator_node)
graph_builder.add_node("invoke_simple", invoke_pipeline_simple_node)
graph_builder.add_node("invoke_intermedio", invoke_pipeline_intermedio_node)
graph_builder.add_node("invoke_complejo", invoke_pipeline_complejo_node)
graph_builder.add_node("validate_decision", validate_decition_node)
graph_builder.add_node("output_node", output_node)

# Cambia el flujo: después de cada pipeline, va a validación

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
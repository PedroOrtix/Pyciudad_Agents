from typing import Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

# Import Pydantic models
from agents.common.schemas import PipelineDecisionSchema

# Import main state for the meta-agent
from agents.Agent_ensemble.states_ensemble import MainAgentState

# Import prompts
from agents.Agent_ensemble.prompt_ensemble import META_AGENT_EVALUATOR_PROMPT

# import llms
from agents.common.llm_config import llm

# Import the compiled subgraph apps from each agent
from agents.Agent_base.agent_base import app_base
from agents.Agent_intention.agent_intention import app_intention
from agents.Agent_validation.agent_validation import app_validation

# --- Meta-Agent Graph (Main Graph) ---
def meta_agent_evaluator_node(state: MainAgentState) -> Dict[str, Any]:
    print("[MainGraph] > Meta-Agent Evaluator")
    user_query = state["user_query"]
    structured_llm = llm.with_structured_output(PipelineDecisionSchema)
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
def invoke_pipeline_simple_node(state: MainAgentState) -> Dict[str, Any]:
    print("[MainGraph] > Invoking PIPELINE_SIMPLE (app_base)")
    subgraph_result = app_base.invoke({"user_query": state["user_query"]})
    return {
        "final_cartociudad_params": subgraph_result.get("cartociudad_query_params"),
        "candidates": subgraph_result.get("final_candidates", [])
    }

def invoke_pipeline_intermedio_node(state: MainAgentState) -> Dict[str, Any]:
    print("[MainGraph] > Invoking PIPELINE_INTERMEDIO (app_intention)")
    subgraph_result = app_intention.invoke({"user_query": state["user_query"]})
    return {
        "final_cartociudad_params": subgraph_result.get("cartociudad_query_params"),
        "candidates": subgraph_result.get("final_candidates", [])
    }

def invoke_pipeline_complejo_node(state: MainAgentState) -> Dict[str, Any]:
    print("[MainGraph] > Invoking PIPELINE_COMPLEJO (app_validation)")
    subgraph_result = app_validation.invoke({"user_query": state["user_query"]})
    return {
        "final_cartociudad_params": subgraph_result.get("final_cartociudad_params"),
        "candidates": subgraph_result.get("final_candidates", [])
    }

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
graph_builder = StateGraph(MainAgentState)

graph_builder.add_node("meta_evaluator", meta_agent_evaluator_node)
graph_builder.add_node("invoke_simple", invoke_pipeline_simple_node)
graph_builder.add_node("invoke_intermedio", invoke_pipeline_intermedio_node)
graph_builder.add_node("invoke_complejo", invoke_pipeline_complejo_node)

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

graph_builder.add_edge("invoke_simple", END)
graph_builder.add_edge("invoke_intermedio", END)
graph_builder.add_edge("invoke_complejo", END)

app_ensemble = graph_builder.compile()
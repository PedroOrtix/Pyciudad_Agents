from typing import Optional, Dict, Any, List
import json
import random

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import Literal

# import pydantic models
from agents.common.schemas import CandidateSchema

# import custom states
from data.Agent_dataset_generator.states_dataset import GraphStateInput, AgentState, GraphStateOutput

# import prompts
from data.Agent_dataset_generator.prompt_dataset import (
    QUERY_GENERATION_PROMPT, 
    ERROR_INJECTION_PROMPT
)

# import LLM configuration
from agents.common.llm_config import llm

# --- Esquemas Pydantic ---

class QueryVariation(BaseModel):
    user_query: str = Field(description="Consulta del usuario generada")
    query_type: Literal["directa", "natural", "coloquial", "pregunta"] = Field(description="Tipo de consulta")
    difficulty_level: Literal["facil", "medio", "alto"] = Field(description="Nivel de dificultad")

class QueryVariationsSchema(BaseModel):
    variations: List[QueryVariation] = Field(description="Lista de variaciones de consultas")

# Tipos de errores fijos predefinidos
ErrorType = Literal[
    "mayusculas_inconsistentes",
    "errores_ortograficos", 
    "espaciado_incorrecto",
    "abreviaciones_incorrectas",
    "tildes_incorrectas",
    "sustituciones_caracteres"
]

class ErrorVariationSchema(BaseModel):
    query_with_errors: str = Field(description="Consulta con errores inyectados")
    error_types: List[ErrorType] = Field(description="Tipos de errores introducidos (debe ser una lista de los tipos predefinidos)")

# --- Node Functions ---

def load_ground_truth_node(state: GraphStateInput) -> Dict[str, Any]:
    """Carga el dataset de ground truth desde el archivo JSON"""
    print("--- Running Node: Load Ground Truth ---")
    
    try:
        with open('data/datasets/dataset_direcciones_init.json', 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)
        
        print(f"Loaded {len(ground_truth_data)} ground truth addresses")
        
        # Seleccionar una muestra aleatoria si se especifica
        if state.sample_size and state.sample_size < len(ground_truth_data):
            selected_data = random.sample(ground_truth_data, state.sample_size)
        else:
            selected_data = ground_truth_data
            
        return {
            "ground_truth_addresses": selected_data,
            "sample_size": state.sample_size,
            "output_filename": state.output_filename,
            "variations_per_address": state.variations_per_address
        }
        
    except Exception as e:
        print(f"Error loading ground truth data: {e}")
        return {
            "ground_truth_addresses": [],
            "sample_size": state.sample_size,
            "output_filename": state.output_filename,
            "variations_per_address": state.variations_per_address
        }

def generate_all_queries_node(state: AgentState) -> Dict[str, Any]:
    """Genera consultas en lenguaje natural para todas las direcciones"""
    print("--- Running Node: Generate All Natural Queries ---")
    
    ground_truth_addresses = state.get("ground_truth_addresses", [])
    all_variations = []
    
    for i, current_address in enumerate(ground_truth_addresses):
        address_text = current_address.get("address", "")
        print(f"Processing address {i+1}/{len(ground_truth_addresses)}: {address_text}")
        
        try:
            structured_llm = llm.with_structured_output(QueryVariationsSchema)
            response = structured_llm.invoke([
                SystemMessage(content=QUERY_GENERATION_PROMPT),
                HumanMessage(content=f"Dirección de referencia: {address_text}")
            ])
            
            # Procesar las variaciones generadas
            for variation in response.variations:
                all_variations.append({
                    "ground_truth_id": current_address.get("id"),
                    "ground_truth_address": address_text,
                    "user_query": variation.user_query,
                    "query_type": variation.query_type,
                    "difficulty_level": variation.difficulty_level,
                    "ground_truth_data": current_address
                })
                
            print(f"Generated {len(response.variations)} variations for: {address_text}")
            
        except Exception as e:
            print(f"Error processing address {address_text}: {e}")
            continue
    
    print(f"Total variations generated: {len(all_variations)}")
    return {
        "generated_variations": all_variations
    }

def inject_errors_node(state: AgentState) -> Dict[str, Any]:
    """Inyecta errores ortográficos y de escritura en las consultas"""
    print("--- Running Node: Inject Errors ---")
    
    generated_variations = state.get("generated_variations", [])
    error_variations = []
    
    for i, variation in enumerate(generated_variations):
        # Mantener la versión original
        variation["has_errors"] = False
        error_variations.append(variation)
        
        # Generar versión con errores para consultas de dificultad media/alta
        if variation["difficulty_level"] in ["medio", "alto"]:
            try:
                structured_llm = llm.with_structured_output(ErrorVariationSchema)
                response = structured_llm.invoke([
                    SystemMessage(content=ERROR_INJECTION_PROMPT),
                    HumanMessage(content=f"Consulta original: {variation['user_query']}")
                ])
                
                # Crear nueva variación con errores
                error_variation = variation.copy()
                error_variation.update({
                    "user_query": response.query_with_errors,
                    "error_types": response.error_types,
                    "original_clean_query": variation["user_query"],
                    "has_errors": True
                })
                error_variations.append(error_variation)
                
            except Exception as e:
                print(f"Error injecting errors for variation {i}: {e}")
                continue
    
    print(f"Generated {len(error_variations)} total variations (with and without errors)")
    
    return {
        "final_dataset": error_variations
    }

def save_dataset_node(state: AgentState) -> Dict[str, Any]:
    """Guarda el dataset generado en un archivo JSON"""
    print("--- Running Node: Save Dataset ---")
    
    final_dataset = state.get("final_dataset", [])
    output_filename = state.get("output_filename", "generated_dataset.json")
    
    try:
        output_path = f"data/datasets/{output_filename}"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(final_dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset saved to {output_path} with {len(final_dataset)} entries")
        
        # Generar estadísticas
        stats = {
            "total_entries": len(final_dataset),
            "with_errors": len([d for d in final_dataset if d.get("has_errors", False)]),
            "without_errors": len([d for d in final_dataset if not d.get("has_errors", False)]),
            "by_difficulty": {
                "facil": len([d for d in final_dataset if d.get("difficulty_level") == "facil"]),
                "medio": len([d for d in final_dataset if d.get("difficulty_level") == "medio"]),
                "alto": len([d for d in final_dataset if d.get("difficulty_level") == "alto"])
            },
            "by_query_type": {}
        }
        
        # Contar por tipos de consulta
        for item in final_dataset:
            query_type = item.get("query_type", "unknown")
            stats["by_query_type"][query_type] = stats["by_query_type"].get(query_type, 0) + 1
        
        return {
            "generated_dataset": final_dataset,
            "dataset_path": output_path,
            "statistics": stats
        }
        
    except Exception as e:
        print(f"Error saving dataset: {e}")
        return {
            "generated_dataset": [],
            "dataset_path": "",
            "statistics": {}
        }

# --- Graph Definition ---
graph_builder = StateGraph(AgentState, input=GraphStateInput, output=GraphStateOutput)

# Add nodes
graph_builder.add_node("load_ground_truth", load_ground_truth_node)
graph_builder.add_node("generate_all_queries", generate_all_queries_node)
graph_builder.add_node("inject_errors", inject_errors_node)
graph_builder.add_node("save_dataset", save_dataset_node)

# Define edges - flujo lineal simple
graph_builder.add_edge(START, "load_ground_truth")
graph_builder.add_edge("load_ground_truth", "generate_all_queries")
graph_builder.add_edge("generate_all_queries", "inject_errors")
graph_builder.add_edge("inject_errors", "save_dataset")
graph_builder.add_edge("save_dataset", END)

# Compile the graph
app_dataset_generator = graph_builder.compile() 
# --- Graph State ---

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

class GraphStateInput(BaseModel):
    sample_size: Optional[int] = Field(default=None, description="Número de direcciones a procesar (None para todas)")
    output_filename: Optional[str] = Field(default="generated_dataset.json", description="Nombre del archivo de salida")
    variations_per_address: Optional[int] = Field(default=5, description="Número de variaciones a generar por dirección")

class AgentState(MessagesState):
    # Datos de entrada
    sample_size: Optional[int] = Field(default=None, description="Número de direcciones a procesar")
    output_filename: Optional[str] = Field(default="generated_dataset.json", description="Nombre del archivo de salida")
    variations_per_address: Optional[int] = Field(default=5, description="Número de variaciones a generar por dirección")
    
    # Datos del ground truth
    ground_truth_addresses: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Lista de direcciones del ground truth")
    
    # Variaciones generadas
    generated_variations: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Variaciones de consultas generadas")
    final_dataset: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Dataset final con errores inyectados")
    
    # Resultados finales
    generated_dataset: Optional[List[Dict[str, Any]]] = Field(default_factory=list, description="Dataset generado completo")
    dataset_path: Optional[str] = Field(default="", description="Ruta donde se guardó el dataset")
    statistics: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Estadísticas del dataset generado")

class GraphStateOutput(BaseModel):
    generated_dataset: List[Dict[str, Any]] = Field(description="Dataset generado completo")
    dataset_path: str = Field(description="Ruta donde se guardó el dataset")
    statistics: Dict[str, Any] = Field(description="Estadísticas del dataset generado") 
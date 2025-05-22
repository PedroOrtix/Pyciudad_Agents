from typing import List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from agents.common.schemas import CartoCiudadQuerySchema, CandidateSchema, ValidationOutput

class GraphStateInput(BaseModel):
    user_query: str

class AgentState(MessagesState):
    user_query: str
    current_cartociudad_params: Optional[CartoCiudadQuerySchema] = None
    last_failed_params: Optional[CartoCiudadQuerySchema] = None
    last_failed_candidates: Optional[List[CandidateSchema]] = Field(default_factory=list)
    last_validation_reflexion: Optional[str] = None 

    candidates_current_iteration: Optional[List[CandidateSchema]] = Field(default_factory=list)
    
    # Lista para acumular TODOS los candidatos de TODAS las iteraciones
    all_candidates_across_iterations: List[CandidateSchema] = Field(default_factory=list, description="Acumula todos los candidatos únicos de todas las iteraciones.")

    validation_output: Optional[ValidationOutput] = None
    reformulation_attempts: int = Field(default=0)

    # Para saber si el flujo terminó por alcanzar el límite de reformulaciones sin éxito
    max_reformulations_reached_with_insufficient_results: bool = False


class GraphStateOutput(BaseModel):
    final_candidates: List[CandidateSchema] = Field(description="The final list of candidates from CartoCiudad.")
    final_cartociudad_params: CartoCiudadQuerySchema = Field(description="The final parameters used for the CartoCiudad query.")
from typing import List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState
from agents.common.schemas import CartoCiudadQuerySchema, CandidateSchema

class GraphStateInput(BaseModel):
    user_query: str = Field(description="The original query from the user.")

# main agent state
class MainAgentState(MessagesState):
    user_query: str
    selected_pipeline_name: Optional[str] = None
    pipeline_justification: Optional[str] = None
    # Estos ahora vendrán directamente del subgrafo
    final_cartociudad_params: Optional[CartoCiudadQuerySchema] = None
    candidates: List[CandidateSchema] = Field(default_factory=list)
    original_candidates: Optional[List[CandidateSchema]] = Field(default_factory=list, description="Candidatos originales obtenidos de la API, usados para reordenar tras el rerank.")
    
class GraphStateOutput(BaseModel):
    final_candidates: List[CandidateSchema] = Field(description="The final list of candidates from CartoCiudad.")
    cartociudad_query_params: Optional[CartoCiudadQuerySchema] = Field(default=None, description="Parameters for CartoCiudad API.")
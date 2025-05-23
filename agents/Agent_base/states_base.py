# --- Graph State ---

from typing import List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from agents.common.schemas import CartoCiudadQuerySchema, CandidateSchema

class GraphStateInput(BaseModel):
    user_query: str = Field(description="The original query from the user.")
    context_from_meta_evaluator: Optional[str] = Field(default=None, description="Context information from the meta-evaluator.")


class AgentState(MessagesState):
    user_query: str = Field(description="The original query from the user.")
    normalized_query: Optional[str] = Field(default=None, description="Normalized user query.")
    keywords: Optional[List[str]] = Field(default_factory=list, description="Extracted keywords.")
    cartociudad_query_params: Optional[CartoCiudadQuerySchema] = Field(default=None, description="Parameters for CartoCiudad API.")
    candidates: Optional[List[CandidateSchema]] = Field(default_factory=list, description="Candidates found by CartoCiudad.")
    context_from_meta_evaluator: Optional[str] = Field(default=None, description="Context information from the meta-evaluator.")

class GraphStateOutput(BaseModel):
    final_candidates: List[CandidateSchema] = Field(description="The final list of candidates from CartoCiudad.")
    cartociudad_query_params: Optional[CartoCiudadQuerySchema] = Field(default=None, description="Parameters for CartoCiudad API.")
# --- Graph State ---
from typing import List, Optional
from pydantic import BaseModel, Field
from langgraph.graph import MessagesState

from agents.common.schemas import CartoCiudadQuerySchema, CandidateSchema, IntentInfo

class GraphStateInput(BaseModel):
    user_query: str = Field(description="The original query from the user.")

class AgentState(MessagesState):
    user_query: str = Field(description="The original query from the user.")
    # Outputs from parallel branches
    keywords: Optional[List[str]] = Field(default_factory=list)
    intent_info: Optional[IntentInfo] = Field(default=None)
    # Output from merged branch
    cartociudad_query_params: Optional[CartoCiudadQuerySchema] = Field(default=None)
    candidates: Optional[List[CandidateSchema]] = Field(default_factory=list)

class GraphStateOutput(BaseModel):
    final_candidates: List[CandidateSchema] = Field(description="The final list of candidates from CartoCiudad.")
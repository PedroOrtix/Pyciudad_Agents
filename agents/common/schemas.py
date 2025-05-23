
from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


# Pydantic Schemas for Agent_base
class NormalizedQueryKeywords(BaseModel):
    """Schema for the output of keyword extraction and normalization."""
    normalized_query: str = Field(description="The user query after normalization and spell checking.")
    keywords: List[str] = Field(description="A list of extracted keywords relevant for the geographic search.")

class CartoCiudadQuerySchema(BaseModel):
    """
    Schema for structuring queries to CartoCiudad.
    """
    consulta: str
    limite: Optional[int] = 10
    municipio: Optional[Union[str, List[str]]] = None
    provincia: Optional[Union[str, List[str]]] = None

class CandidateSchema(BaseModel):
    """
    Simplified model to store only relevant fields of a candidate.
    Mimics the one from agent_0.py for consistency.
    """
    id: Optional[str] = Field(default=None, description="Unique identifier of the entity")
    type: Optional[str] = Field(default=None, description="Entity type (e.g.: callejero, toponimo)")
    address: Optional[str] = Field(default=None, description="Address or name of the entity")
    model_config = ConfigDict(extra="ignore")

# Pydantic Schemas for Agent_intetion

# NormalizedQueryKeywords, CartoCiudadQuerySchema, CandidateSchema from agent_base.py

class IntentInfo(BaseModel):
    """Schema for the output of intent detection."""
    intent: str = Field(description="Intentions detected in the user query.")
    justification: str = Field(description="A brief justification for the detected intent and entity type.")
    
# Pydantic Schemas for Agent_intetion

# CartoCiudadQuerySchema, CandidateSchema from agent_base.py

class ValidationOutput(BaseModel):
    reflexion_interna: str
    decision_final: Literal["Suficiente", "Necesita_Reformulacion"]
    
# Pydantic Schemas for Agent_ensemble

class PipelineDecisionSchema(BaseModel):
    justification: str
    selected_pipeline: Literal["PIPELINE_SIMPLE", "PIPELINE_INTERMEDIO", "PIPELINE_COMPLEJO"]

# Pydantic Schemas for Agent_fallback
class QualityDecision(BaseModel):
    justification: str
    decision: Literal["Suficiente", "Insuficiente_Escalar"]
    
class RerankSchema(BaseModel):
    """
    Schema for the output of the reranking process.
    """
    rerank_candidates: List[CandidateSchema] = Field(description="List of candidates after reranking.")
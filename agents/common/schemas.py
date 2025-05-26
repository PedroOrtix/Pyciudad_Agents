from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict


# Pydantic Schemas for Agent_base
class NormalizedQueryKeywords(BaseModel):
    normalized_query: str = Field(description="Consulta normalizada del usuario.")
    keywords: List[str] = Field(description="Palabras clave extraídas relevantes para la búsqueda geográfica.")

class CartoCiudadQuerySchema(BaseModel):
    consulta: str
    limite: Optional[int] = 10
    municipio: Optional[Union[str, List[str]]] = None
    provincia: Optional[Union[str, List[str]]] = None

class CandidateSchema(BaseModel):
    id: Optional[str] = Field(default=None, description="Identificador único de la entidad")
    type: Optional[str] = Field(default=None, description="Tipo de entidad (ej: callejero, toponimo)")
    address: Optional[str] = Field(default=None, description="Dirección o nombre de la entidad")
    model_config = ConfigDict(extra="ignore")

# Pydantic Schemas for Agent_intetion

# NormalizedQueryKeywords, CartoCiudadQuerySchema, CandidateSchema from agent_base.py

class IntentInfo(BaseModel):
    intent: str = Field(description="Intención detectada en la consulta del usuario.")
    justification: str = Field(description="Justificación breve para la intención detectada.")
    
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
    rerank_candidates: List[CandidateSchema] = Field(description="Lista de candidatos tras el reranking.")

class RerankOrderSchema(BaseModel):
    ordered_ids: List[str] = Field(description="Lista de IDs de candidatos en el orden deseado tras el reranking.")
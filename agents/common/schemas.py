from typing import List, Optional, Union, Literal
from pydantic import BaseModel, Field, ConfigDict

# --- Esquemas para Agent_base ---

class NormalizedQueryKeywords(BaseModel):
    """Esquema para consulta normalizada y palabras clave extraídas."""
    normalized_query: str = Field(description="Consulta del usuario normalizada y limpia")
    keywords: List[str] = Field(description="Palabras clave relevantes para búsqueda geográfica")

class CartoCiudadQuerySchema(BaseModel):
    """Esquema para parámetros de consulta a la API de CartoCiudad."""
    consulta: str = Field(description="Término de búsqueda principal")
    limite: Optional[int] = Field(default=10, description="Número máximo de resultados")
    municipio: Optional[Union[str, List[str]]] = Field(default=None, description="Filtro por municipio(s)")
    provincia: Optional[Union[str, List[str]]] = Field(default=None, description="Filtro por provincia(s)")

class CandidateSchema(BaseModel):
    """Esquema para candidatos de ubicación devueltos por CartoCiudad."""
    id: Optional[str] = Field(default=None, description="Identificador único de la entidad")
    type: Optional[str] = Field(default=None, description="Tipo de entidad (ej: callejero, toponimo)")
    address: Optional[str] = Field(default=None, description="Dirección o nombre de la entidad")
    model_config = ConfigDict(extra="ignore")

# --- Esquemas para Agent_intention ---

class IntentInfo(BaseModel):
    """Esquema para información de intención detectada en la consulta."""
    intent: str = Field(description="Tipo de intención detectada en la consulta")
    justification: str = Field(description="Justificación para la intención detectada")
    
# --- Esquemas para Agent_validation ---

class ValidationOutput(BaseModel):
    """Esquema para salida del agente validador."""
    reflexion_interna: str = Field(description="Análisis interno sobre la calidad de los resultados")
    decision_final: Literal["Suficiente", "Necesita_Reformulacion"] = Field(description="Decisión sobre si los resultados son adecuados")
    
# --- Esquemas para Agent_ensemble ---

class PipelineDecisionSchema(BaseModel):
    """Esquema para decisión de selección de pipeline."""
    justification: str = Field(description="Justificación para la selección del pipeline")
    selected_pipeline: Literal["PIPELINE_SIMPLE", "PIPELINE_INTERMEDIO", "PIPELINE_COMPLEJO"] = Field(description="Pipeline seleccionado")

# --- Esquemas compartidos para reranking ---

class RerankSchema(BaseModel):
    """Esquema para candidatos reordenados (esquema legacy)."""
    rerank_candidates: List[CandidateSchema] = Field(description="Lista de candidatos tras el reranking")

class RerankOrderSchema(BaseModel):
    """Esquema para orden de IDs de candidatos tras reranking."""
    ordered_ids: List[str] = Field(description="Lista de IDs de candidatos en orden de relevancia")

# --- Esquemas adicionales ---

class QualityDecision(BaseModel):
    """Esquema para decisiones de calidad (usado en agentes de fallback)."""
    justification: str = Field(description="Justificación para la decisión de calidad")
    decision: Literal["Suficiente", "Insuficiente_Escalar"] = Field(description="Decisión sobre calidad de resultados")
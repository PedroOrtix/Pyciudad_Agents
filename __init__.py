"""
PyCiudad Agents - Sistema de Evaluación

Framework comprehensivo para la evaluación de agentes LLM
especializados en tareas de geolocalización.
"""

# Re-export principales del sistema de evaluación
from evaluation import (
    AgentEvaluationSystem,
    GroundTruthEvaluator,
    evaluate_single_execution,
    get_agent_model_config,
    get_model_pricing,
    get_model_price,
    update_evaluation_config
)

__version__ = "1.0.0"

# Facilitar importaciones comunes
__all__ = [
    "AgentEvaluationSystem",
    "GroundTruthEvaluator",
    "evaluate_single_execution",
    "get_agent_model_config",
    "get_model_pricing",
    "get_model_price",
    "update_evaluation_config"
] 
"""
Sistema de Evaluación PyCiudad con Ground Truth

Módulo comprehensivo para evaluar agentes de geolocalización incluyendo:
- Evaluación técnica (robustez)
- Evaluación de calidad (ground truth)
- Análisis comparativo entre configuraciones
- Métricas avanzadas y visualizaciones

NUEVA FUNCIONALIDAD:
- GroundTruthEvaluator: Evaluación de calidad basada en posición
- Sistema de scoring: 1.0 (pos.1), 0.8 (top-3), 0.6 (top-5), 0.3 (>5), 0.0 (no encontrado)
- Nueva definición de éxito: técnico + calidad
"""

# Core evaluation system
from evaluation.core.agent_evaluation_system import AgentEvaluationSystem

from evaluation.core.ground_truth_evaluator import (
    GroundTruthEvaluator, 
    evaluate_single_execution
)

# Analysis and metrics
# from evaluation.analysis.metrics_analyzer import MetricsAnalyzer  # TODO: Implementar
# from evaluation.analysis.compare_model_configs import ComprehensiveModelComparator  # TODO: Implementar

# Configuration management
from evaluation.configs.config_loader import (
    get_agent_model_config,
    get_model_pricing,
    get_model_price,
    update_evaluation_config
)

__version__ = "1.0.0"
__author__ = "PyCiudad Team"

# Exports principales
__all__ = [
    # Core system
    "AgentEvaluationSystem",
    
    # Ground truth evaluation (NUEVO)
    "GroundTruthEvaluator",
    "evaluate_single_execution",
    
    # Configuration
    "get_agent_model_config",
    "get_model_pricing", 
    "get_model_price",
    "update_evaluation_config"
] 
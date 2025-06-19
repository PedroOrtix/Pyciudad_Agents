"""
Evaluador de Ground Truth para Resultados de Agentes

Este módulo implementa la evaluación real de calidad de los resultados
comparándolos con el ground truth del dataset, usando un sistema de 
scoring basado en la posición del resultado correcto.

Sistema de Scoring:
- Posición 1: 1.0 puntos (encontrado en primer lugar)
- Posiciones 1-3: 0.8 puntos (encontrado en top 3)
- Posiciones 1-5: 0.6 puntos (encontrado en top 5)
- Más de 5 posiciones: 0.3 puntos (encontrado pero lejos)
- No encontrado: 0.0 puntos
"""

from typing import Dict, List, Any, Optional


class GroundTruthEvaluator:
    """Evaluador de calidad basado en ground truth"""
    
    def __init__(self):
        """Inicializar el evaluador"""
        self.scoring_rules = {
            1: 1.0,      # Posición exacta
            3: 0.8,      # Top 3
            5: 0.6,      # Top 5
            999: 0.3,    # Encontrado pero fuera del top 5
            0: 0.0       # No encontrado
        }
    
    def evaluate_agent_result(self, 
                            agent_output: Dict[str, Any],
                            ground_truth: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluar un resultado individual del agente contra ground truth
        
        Args:
            agent_output: Salida del agente con final_candidates
            ground_truth: Información de ground truth del sample
            
        Returns:
            Dict con métricas de evaluación
        """
        
        # Extraer el ground truth ID
        ground_truth_id = ground_truth.get("ground_truth_id")
        if not ground_truth_id:
            return {
                "quality_score": 0.0,
                "position_found": None,
                "total_candidates": 0,
                "evaluation_error": "No ground_truth_id available"
            }
        
        # Extraer candidatos del output del agente
        candidates = self._extract_candidates_from_output(agent_output)
        if not candidates:
            return {
                "quality_score": 0.0,
                "position_found": None,
                "total_candidates": 0,
                "ground_truth_id": ground_truth_id,
                "found_in_results": False,
                "scoring_tier": "not_found",
                "evaluation_error": "No candidates in agent output"
            }
        
        # Buscar la posición del ground truth en los resultados
        position = self._find_ground_truth_position(candidates, ground_truth_id)
        
        # Calcular score basado en posición
        quality_score = self._calculate_position_score(position)
        
        return {
            "quality_score": quality_score,
            "position_found": position,
            "total_candidates": len(candidates),
            "ground_truth_id": ground_truth_id,
            "found_in_results": position is not None,
            "scoring_tier": self._get_scoring_tier(position)
        }
    
    def _extract_candidates_from_output(self, agent_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extraer lista de candidatos del output del agente"""
        
        # El output puede estar en diferentes estructuras
        if "final_candidates" in agent_output:
            candidates = agent_output["final_candidates"]
        elif "candidates" in agent_output:
            candidates = agent_output["candidates"]
        else:
            # Buscar en estructuras anidadas
            candidates = []
            
        # Convertir candidatos a lista de diccionarios si es necesario
        if isinstance(candidates, list):
            result = []
            for candidate in candidates:
                if hasattr(candidate, 'model_dump'):
                    result.append(candidate.model_dump())
                elif isinstance(candidate, dict):
                    result.append(candidate)
                else:
                    result.append({"id": str(candidate)})
            return result
        
        return []
    
    def _find_ground_truth_position(self, 
                                  candidates: List[Dict[str, Any]], 
                                  ground_truth_id: str) -> Optional[int]:
        """
        Encontrar la posición del ground truth en la lista de candidatos
        
        Returns:
            Posición (1-indexed) si se encuentra, None si no
        """
        for i, candidate in enumerate(candidates):
            candidate_id = candidate.get("id")
            if candidate_id and str(candidate_id) == str(ground_truth_id):
                return i + 1  # 1-indexed position
        
        return None
    
    def _calculate_position_score(self, position: Optional[int]) -> float:
        """
        Calcular score basado en la posición del resultado correcto
        
        Args:
            position: Posición encontrada (1-indexed) o None
            
        Returns:
            Score según las reglas definidas
        """
        if position is None:
            return 0.0
        
        if position == 1:
            return 1.0
        elif position <= 3:
            return 0.8
        elif position <= 5:
            return 0.6
        else:
            return 0.3
    
    def _get_scoring_tier(self, position: Optional[int]) -> str:
        """Obtener el tier de scoring para análisis"""
        if position is None:
            return "not_found"
        elif position == 1:
            return "perfect"
        elif position <= 3:
            return "top_3"
        elif position <= 5:
            return "top_5"
        else:
            return "found_far"
    
    def evaluate_batch_results(self, 
                             results: List[Dict[str, Any]],
                             ground_truths: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluar un lote de resultados
        
        Args:
            results: Lista de resultados de agentes
            ground_truths: Lista de ground truths correspondientes
            
        Returns:
            Estadísticas agregadas del lote
        """
        evaluations = []
        
        for result, gt in zip(results, ground_truths):
            evaluation = self.evaluate_agent_result(result, gt)
            evaluations.append(evaluation)
        
        # Calcular estadísticas agregadas
        total_samples = len(evaluations)
        if total_samples == 0:
            return {"error": "No samples to evaluate"}
        
        # Calcular métricas agregadas
        quality_scores = [e["quality_score"] for e in evaluations]
        positions_found = [e["position_found"] for e in evaluations if e["position_found"]]
        
        # Contar por tiers de scoring
        tier_counts = {}
        for evaluation in evaluations:
            tier = evaluation["scoring_tier"]
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
        
        # Calcular métricas finales
        stats = {
            "total_samples": total_samples,
            "average_quality_score": sum(quality_scores) / total_samples,
            "found_rate": len(positions_found) / total_samples,
            "perfect_rate": tier_counts.get("perfect", 0) / total_samples,
            "top_3_rate": (tier_counts.get("perfect", 0) + tier_counts.get("top_3", 0)) / total_samples,
            "top_5_rate": (tier_counts.get("perfect", 0) + tier_counts.get("top_3", 0) + tier_counts.get("top_5", 0)) / total_samples,
            "tier_distribution": tier_counts,
            "position_stats": {
                "min_position": min(positions_found) if positions_found else None,
                "max_position": max(positions_found) if positions_found else None,
                "avg_position": sum(positions_found) / len(positions_found) if positions_found else None
            }
        }
        
        return stats


def evaluate_single_execution(agent_output: Any, ground_truth_sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Función auxiliar para evaluar una sola ejecución
    
    Args:
        agent_output: Output del agente (puede ser dict o object)
        ground_truth_sample: Sample del dataset con ground truth
        
    Returns:
        Evaluación de calidad
    """
    evaluator = GroundTruthEvaluator()
    
    # Convertir agent_output a dict si es necesario
    if hasattr(agent_output, 'model_dump'):
        agent_output_dict = agent_output.model_dump()
    elif hasattr(agent_output, '__dict__'):
        agent_output_dict = agent_output.__dict__
    elif isinstance(agent_output, dict):
        agent_output_dict = agent_output
    else:
        agent_output_dict = {"candidates": []}
    
    return evaluator.evaluate_agent_result(agent_output_dict, ground_truth_sample) 
"""
Sistema de Evaluación de Agentes PyCiudad LOCAL

Sistema que ejecuta agentes localmente sin depender del servicio LangGraph
para evitar problemas de rate limiting.
"""

import asyncio
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
import numpy as np

# Importamos directamente los agentes compilados
from agents.Agent_base.agent_base import app_base
from agents.Agent_intention.agent_intention import app_intention  
from agents.Agent_validation.agent_validation import app_validation
from agents.Agent_ensemble.agent_ensemble import app_ensemble

# Ground truth evaluation
from evaluation.core.ground_truth_evaluator import GroundTruthEvaluator, evaluate_single_execution
from evaluation.core.network_resilience import (
    with_network_resilience, 
    EvaluationCheckpoint, 
    NetworkError,
    test_connectivity
)


def numpy_json_serializer(obj):
    """Serializador personalizado para manejar tipos numpy y Pydantic"""
    # Manejar objetos Pydantic
    if hasattr(obj, 'model_dump'):
        return obj.model_dump()
    
    if hasattr(obj, 'dtype'):
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # Fallback para tipos específicos de numpy
    if str(type(obj)).startswith('<class \'numpy.'):
        if 'int' in str(type(obj)):
            return int(obj)
        elif 'float' in str(type(obj)):
            return float(obj)
        elif 'bool' in str(type(obj)):
            return bool(obj)
    # Manejar métodos y funciones (no serializar)
    if callable(obj):
        return f"<callable: {str(obj)}>"
    # Manejar sets
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


class AgentEvaluationSystemLocal:
    """Sistema de evaluación LOCAL para todos los agentes de PyCiudad"""
    
    def __init__(self, 
                 dataset_path: str = "data/datasets/dataset_generado_final.json"):
        """
        Inicializar el sistema de evaluación local
        
        Args:
            dataset_path: Ruta al dataset con ground truth
        """
        self.dataset_path = dataset_path
        
        # Definir agentes disponibles y sus aplicaciones compiladas
        self.agents = {
            "agent_base": app_base,
            "agent_intention": app_intention,
            "agent_validation": app_validation,
            "agent_ensemble": app_ensemble
        }
        
        # Capturar configuración de modelos al momento de la evaluación
        self.model_config = self._capture_model_config()
        
        # Inicializar evaluador de ground truth
        self.ground_truth_evaluator = GroundTruthEvaluator()
        
        # Inicializar sistema de checkpoints
        self.checkpoint_system = EvaluationCheckpoint()
        
        print("🎯 Sistema de Evaluación LOCAL inicializado")
        print(f"📊 Dataset: {dataset_path}")
        print(f"🤖 Agentes disponibles: {', '.join(self.agents.keys())}")
        print("📈 Evaluación con Ground Truth: ACTIVADA")
        print("🔧 Modo: EJECUCIÓN LOCAL (sin LangGraph server)")
        print("🛡️  Sistema de robustez de red: ACTIVADO")
        
        # Verificar conectividad inicial
        try:
            test_connectivity()
        except Exception as e:
            print(f"⚠️  Advertencia: Problema inicial de conectividad: {e}")
    
    def _capture_model_config(self) -> Dict[str, str]:
        """Capturar configuración actual de modelos desde variables de entorno"""
        config = {}
        
        # Variables de entorno comunes para modelos
        env_vars = [
            "OPENAI_MODEL_NAME", "ANTHROPIC_MODEL_NAME", "OLLAMA_MODEL", 
            "OLLAMA_MODEL_THINKING", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"
        ]
        
        for var in env_vars:
            value = os.getenv(var)
            if value:
                config[var] = value
        
        # Timestamp de captura
        config["captured_at"] = datetime.now().isoformat()
        
        return config

    def load_dataset(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Cargar dataset con ground truth"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            if max_samples:
                dataset = dataset[:max_samples]
                
            print("📊 Dataset cargado: {len(dataset)} muestras")
            
            # Verificar que todas las muestras tienen ground truth
            missing_gt = [i for i, sample in enumerate(dataset) if not sample.get("ground_truth_id")]
            if missing_gt:
                print("⚠️  {len(missing_gt)} muestras sin ground_truth_id")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error cargando dataset: {e}")
            return []

    @with_network_resilience(max_retries=3, base_delay=1.0, connectivity_wait_minutes=10)
    async def execute_agent_on_sample_local(self, agent_config: Dict[str, Any], sample: Dict[str, Any], 
                                          sample_index: int, evaluation_id: str) -> Dict[str, Any]:
        """Ejecutar un agente en una muestra específica usando ejecución local"""
        
        agent_name = agent_config.get("name")
        agent_app = agent_config.get("app")
        
        # Obtener datos de ground truth
        ground_truth_id = sample.get("ground_truth_id")
        
        # Inicializar variables para captura de errores
        run_status = "failed"
        agent_output = {}
        error = None
        
        start_time = time.time()
        
        try:
            print(f"🤖 Ejecutando {agent_name} en muestra {sample_index + 1}... (LOCAL)")
            
            # Preparar input para el agente
            agent_input = {
                'user_query': sample.get('user_query', ''),
                'context_from_meta_evaluator': None
            }
            
            # Ejecutar el agente LOCAL directamente
            result = agent_app.invoke(agent_input)
            
            # Procesar el resultado basado en el tipo de retorno
            if hasattr(result, 'model_dump'):
                # Si es un objeto Pydantic, usar model_dump()
                agent_output = result.model_dump()
            elif hasattr(result, 'keys') and hasattr(result, '__getitem__'):
                # Si es dict-like (como AddableValuesDict), convertir a dict
                agent_output = dict(result)
            elif hasattr(result, '__dict__'):
                # Si es un objeto con atributos, convertir a dict
                agent_output = result.__dict__
            elif isinstance(result, dict):
                # Si ya es un dict, usar directamente
                agent_output = result
            else:
                # Fallback: intentar extraer información básica
                agent_output = {
                    "raw_result": str(result),
                    "type": str(type(result))
                }
            
            run_status = "completed"
            
        except NetworkError as e:
            print(f"🌐 Error de red ejecutando {agent_name}: {str(e)}")
            error = f"Network error: {str(e)}"
            agent_output = {
                "error": error,
                "user_query": sample.get('user_query', ''),
                "error_type": "network"
            }
            # Propagar NetworkError para manejo a nivel superior
            raise
        except Exception as e:
            print(f"❌ Error ejecutando {agent_name}: {str(e)}")
            error = str(e)
            agent_output = {
                "error": str(e),
                "user_query": sample.get('user_query', ''),
                "error_type": "general"
            }
            
        execution_time = time.time() - start_time
        
        # Extraer candidatos finales
        final_candidates = []
        if "final_candidates" in agent_output:
            final_candidates = agent_output.get("final_candidates", [])
        elif "candidates" in agent_output:
            final_candidates = agent_output.get("candidates", [])
        
        print(f"🔍 final_candidates found: {len(final_candidates)} items")
        
        # Evaluación de calidad con Ground Truth
        quality_metrics = evaluate_single_execution(
            agent_output, 
            sample
        )
        
        # Determinar éxito técnico
        technical_success = run_status == "completed" and error is None
        
        # Determinar éxito combinado (técnico + calidad)
        combined_success = technical_success and quality_metrics.get("found_in_results", False)
        
        # Extraer información rica del dataset
        dataset_info = {
            "query_type": sample.get("query_type"),
            "difficulty_level": sample.get("difficulty_level"),
            "has_errors": sample.get("has_errors", False),
            "error_types": sample.get("error_types", []),
            "original_clean_query": sample.get("original_clean_query")
        }
        
        # Agregar información de ground truth más detallada
        ground_truth_data = sample.get("ground_truth_data", {})
        detailed_ground_truth = {
            "ground_truth_id": ground_truth_id,
            "ground_truth_address": sample.get("ground_truth_address"),
            "ground_truth_coordinates": {
                "lat": ground_truth_data.get("lat"),
                "lng": ground_truth_data.get("lng")
            } if ground_truth_data.get("lat") and ground_truth_data.get("lng") else None,
            "ground_truth_location_info": {
                "municipality": ground_truth_data.get("muni"),
                "province": ground_truth_data.get("province"),
                "autonomous_community": ground_truth_data.get("comunidadAutonoma"),
                "postal_code": ground_truth_data.get("postalCode"),
                "type": ground_truth_data.get("type"),
                "tip_via": ground_truth_data.get("tip_via")
            } if ground_truth_data else None
        }
        
        # Resultado unificado con información rica
        result_data = {
            "agent_name": agent_name,
            "sample_id": sample.get("id"),
            "input_query": sample.get('user_query', ''),
            "execution_time_seconds": execution_time,
            "run_status": run_status,
            "technical_success": technical_success,
            
            # Información rica del dataset (NUEVA)
            "dataset_info": dataset_info,
            
            # Información detallada de ground truth (MEJORADA)
            "ground_truth_info": detailed_ground_truth,
            
            # Métricas de calidad
            "quality_score": quality_metrics.get("quality_score", 0.0),
            "position_found": quality_metrics.get("position_found"),
            "total_candidates": quality_metrics.get("total_candidates", 0),
            "found_in_results": quality_metrics.get("found_in_results", False),
            "scoring_tier": quality_metrics.get("scoring_tier", "not_found"),
            
            # Métricas combinadas
            "combined_success": combined_success,
            "success": combined_success,  # Nueva definición unificada
            
            "timestamp": datetime.now().isoformat(),
            "agent_output": agent_output,
            "metadata": {
                "model_config": self.model_config,
                "sample_index": sample_index,
                "evaluation_id": evaluation_id,
                "network_resilience": True
            }
        }
        
        # Agregar información de error si existe
        if error:
            result_data["error"] = error
            
        # Log detallado del resultado con información rica
        status_icon = "✅" if combined_success else "❌"
        tech_icon = "✓" if technical_success else "✗"
        quality_score = quality_metrics.get("quality_score", 0.0)
        position = quality_metrics.get("position_found", "None")
        sample_id = sample.get("id", "None")
        query_type = dataset_info.get("query_type", "unknown")
        difficulty = dataset_info.get("difficulty_level", "unknown")
        has_errors = "🔴" if dataset_info.get("has_errors") else "🟢"
        
        print(f"{status_icon} {agent_name} - {execution_time:.2f}s - Tech: {tech_icon} - Quality: {quality_score:.2f} - Position: {position} - Type: {query_type} - Difficulty: {difficulty} - Errors: {has_errors} - Sample: {sample_id}")
        
        return result_data

    async def evaluate_agent_on_dataset_local(self, 
                                            agent_name: str, 
                                            dataset: List[Dict[str, Any]],
                                            max_concurrent: int = 3,
                                            evaluation_id: str = None) -> List[Dict[str, Any]]:
        """
        Evaluar un agente específico en el dataset usando ejecución local
        
        Args:
            agent_name: Nombre del agente a evaluar
            dataset: Lista de muestras del dataset
            max_concurrent: Número máximo de ejecuciones concurrentes
            evaluation_id: ID de evaluación para checkpoints
        
        Returns:
            Lista de resultados de evaluación
        """
        print(f"\n🤖 Evaluando agente: {agent_name} (LOCAL)")
        print(f"📊 Muestras a procesar: {len(dataset)}")
        
        if agent_name not in self.agents:
            raise ValueError(f"Agente {agent_name} no disponible. Agentes: {list(self.agents.keys())}")
        
        agent_app = self.agents[agent_name]
        
        # Crear configuración del agente
        agent_config = {
            "name": agent_name,
            "app": agent_app
        }
        
        # Generar ID de evaluación si no se proporciona
        if evaluation_id is None:
            evaluation_id = f"{agent_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Verificar si hay checkpoint previo
        checkpoint_data = self.checkpoint_system.load_latest_checkpoint(evaluation_id)
        completed_samples = set()
        results = []
        
        if checkpoint_data:
            print(f"📂 Reanudando desde checkpoint: {len(checkpoint_data['data'].get('results', []))} muestras completadas")
            results = checkpoint_data['data'].get('results', [])
            completed_samples = {r.get('metadata', {}).get('sample_index', -1) for r in results}
        
        # Crear semáforo para controlar concurrencia
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def execute_with_semaphore(sample, index):
            # Saltar si ya está completada
            if index in completed_samples:
                print(f"⏭️  Saltando muestra {index + 1} (ya completada)")
                return None
                
            async with semaphore:
                try:
                    result = await self.execute_agent_on_sample_local(
                        agent_config, sample, index, evaluation_id
                    )
                    
                    # Guardar checkpoint cada 10 muestras
                    if (len(results) + 1) % 10 == 0:
                        checkpoint_data = {
                            'agent_name': agent_name,
                            'total_samples': len(dataset),
                            'completed_samples': len(results) + 1,
                            'results': results + [result]
                        }
                        self.checkpoint_system.save_checkpoint(evaluation_id, checkpoint_data)
                    
                    return result
                except NetworkError as e:
                    print(f"🌐 Error de red crítico en muestra {index + 1}: {e}")
                    # Guardar checkpoint de emergencia
                    emergency_data = {
                        'agent_name': agent_name,
                        'total_samples': len(dataset),
                        'completed_samples': len(results),
                        'results': results,
                        'last_error': str(e),
                        'failed_sample_index': index
                    }
                    self.checkpoint_system.save_checkpoint(f"{evaluation_id}_emergency", emergency_data)
                    raise
        
        # Ejecutar evaluación concurrente solo en muestras no completadas
        pending_tasks = [
            (i, sample) for i, sample in enumerate(dataset) 
            if i not in completed_samples
        ]
        
        print(f"📋 Procesando {len(pending_tasks)} muestras restantes...")
        
        tasks = [
            execute_with_semaphore(sample, i) 
            for i, sample in pending_tasks
        ]
        
        if tasks:  # Solo si hay tareas pendientes
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Procesar resultados y manejar excepciones
            for i, result in enumerate(task_results):
                if result is None:
                    continue  # Muestra ya completada
                    
                if isinstance(result, Exception):
                    sample_index = pending_tasks[i][0]
                    print(f"❌ Excepción en muestra {sample_index}: {result}")
                    error_result = {
                        "agent_name": agent_name,
                        "sample_id": dataset[sample_index].get("id"),
                        "error": str(result),
                        "timestamp": datetime.now().isoformat(),
                        "success": False,
                        "technical_success": False,
                        "quality_score": 0.0,
                        "combined_success": False,
                        "metadata": {
                            "sample_index": sample_index,
                            "evaluation_id": evaluation_id,
                            "error_type": "exception"
                        }
                    }
                    results.append(error_result)
                else:
                    results.append(result)
        
        # Guardar checkpoint final
        final_data = {
            'agent_name': agent_name,
            'total_samples': len(dataset),
            'completed_samples': len(results),
            'results': results,
            'status': 'completed'
        }
        self.checkpoint_system.save_checkpoint(f"{evaluation_id}_final", final_data)
        
        # Limpiar checkpoints antiguos
        self.checkpoint_system.cleanup_old_checkpoints(evaluation_id)
        
        print(f"✅ {agent_name} completado: {len(results)} resultados")
        return results

    async def evaluate_all_agents_local(self, 
                                      max_samples: Optional[int] = 100,
                                      max_concurrent: int = 3,
                                      agents_to_evaluate: Optional[List[str]] = None,
                                      evaluation_id: Optional[str] = None,
                                      network_wait_minutes: int = 30,
                                      max_retries: int = 5) -> Dict[str, Any]:
        """
        Evaluar todos los agentes en el dataset usando ejecución local
        
        Args:
            max_samples: Número máximo de muestras del dataset (None para todo)
            max_concurrent: Número máximo de ejecuciones concurrentes por agente
            agents_to_evaluate: Lista de agentes a evaluar (None para todos)
            evaluation_id: ID de evaluación para reanudar desde checkpoint
            network_wait_minutes: Minutos a esperar por conectividad
            max_retries: Número máximo de reintentos ante fallos de red
        
        Returns:
            Resultados completos de la evaluación con métricas de calidad
        """
        print("🚀 Iniciando evaluación completa de agentes LOCAL")
        
        # Cargar dataset
        dataset = self.load_dataset(max_samples)
        
        # Determinar qué agentes evaluar
        if agents_to_evaluate is None:
            agents_to_evaluate = list(self.agents.keys())
        
        # Validar que los agentes existen
        invalid_agents = [a for a in agents_to_evaluate if a not in self.agents]
        if invalid_agents:
            raise ValueError(f"Agentes inválidos: {invalid_agents}. Disponibles: {list(self.agents.keys())}")
        
        print(f"🎯 Agentes a evaluar: {', '.join(agents_to_evaluate)}")
        
        # Evaluar cada agente
        all_results = {}
        total_start_time = time.time()
        
        for agent_name in agents_to_evaluate:
            agent_start_time = time.time()
            
            try:
                # Generar ID específico para este agente si se proporciona uno base
                agent_evaluation_id = evaluation_id
                if evaluation_id and len(agents_to_evaluate) > 1:
                    agent_evaluation_id = f"{evaluation_id}_{agent_name}"
                
                agent_results = await self.evaluate_agent_on_dataset_local(
                    agent_name, dataset, max_concurrent, agent_evaluation_id
                )
                all_results[agent_name] = agent_results
                
                agent_time = time.time() - agent_start_time
                print(f"⏱️  {agent_name}: {agent_time:.2f}s")
                
            except NetworkError as e:
                print(f"🌐 Error de red crítico evaluando {agent_name}: {e}")
                print("💾 Checkpoints guardados para reanudar evaluación")
                # Continuar con otros agentes si es posible
                all_results[agent_name] = []
            except Exception as e:
                print(f"❌ Error evaluando {agent_name}: {e}")
                all_results[agent_name] = []
        
        total_time = time.time() - total_start_time
        
        # Generar estadísticas resumidas
        evaluation_summary = self._generate_evaluation_summary(all_results, total_time)
        
        # Agregar información de robustez de red a los metadatos
        final_results = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time_seconds": total_time,
                "dataset_size": len(dataset),
                "agents_evaluated": agents_to_evaluate,
                "execution_mode": "LOCAL",
                "model_config": self.model_config,
                "evaluation_type": "ground_truth_based",
                "network_resilience_config": {
                    "enabled": True,
                    "max_retries": max_retries,
                    "network_wait_minutes": network_wait_minutes,
                    "checkpoint_system": True
                },
                "scoring_system": {
                    "position_1": 1.0,
                    "positions_1_3": 0.8,
                    "positions_1_5": 0.6,
                    "beyond_5": 0.3,
                    "not_found": 0.0
                }
            },
            "results_by_agent": all_results,
            "evaluation_summary": evaluation_summary
        }
        
        print("\n🎉 Evaluación LOCAL completada")
        print(f"⏱️  Tiempo total: {total_time:.2f}s")
        print(f"📊 Agentes evaluados: {len(agents_to_evaluate)}")
        print("📈 Muestras procesadas: {len(dataset)}")
        print("🛡️  Sistema de robustez de red usado: ✅")
        
        return final_results

    def _generate_evaluation_summary(self, all_results: Dict[str, List[Dict]], total_time: float) -> Dict[str, Any]:
        """Generar resumen estadístico de la evaluación"""
        summary = {
            "total_execution_time_seconds": total_time,
            "agents_summary": {}
        }
        
        for agent_name, results in all_results.items():
            if not results:
                continue
                
            # Filtrar resultados válidos
            valid_results = [r for r in results if isinstance(r, dict)]
            
            # Métricas técnicas
            technical_successes = [r for r in valid_results if r.get("technical_success", False)]
            
            # Métricas de calidad
            quality_scores = [r.get("quality_score", 0.0) for r in valid_results]
            found_results = [r for r in valid_results if r.get("found_in_results", False)]
            perfect_hits = [r for r in valid_results if r.get("scoring_tier") == "perfect"]
            top_3_hits = [r for r in valid_results if r.get("scoring_tier") in ["perfect", "top_3"]]
            top_5_hits = [r for r in valid_results if r.get("scoring_tier") in ["perfect", "top_3", "top_5"]]
            
            # Métricas combinadas
            combined_successes = [r for r in valid_results if r.get("combined_success", False)]
            
            # Tiempos de ejecución
            execution_times = [r.get("execution_time_seconds", 0) for r in valid_results if r.get("execution_time_seconds")]
            
            # Posiciones encontradas (solo para resultados exitosos)
            positions = [r.get("position_found") for r in valid_results if r.get("position_found") is not None]
            
            # Distribución por tier
            tier_counts = {}
            for result in valid_results:
                tier = result.get("scoring_tier", "unknown")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            # NUEVO: Análisis por tipo de consulta
            query_type_analysis = {}
            query_types = set()
            for result in valid_results:
                query_type = result.get("dataset_info", {}).get("query_type")
                if query_type:
                    query_types.add(query_type)
            
            for query_type in query_types:
                type_results = [r for r in valid_results 
                              if r.get("dataset_info", {}).get("query_type") == query_type]
                type_successes = [r for r in type_results if r.get("combined_success", False)]
                
                query_type_analysis[query_type] = {
                    "total_samples": len(type_results),
                    "successful": len(type_successes),
                    "success_rate": len(type_successes) / len(type_results) if type_results else 0,
                    "avg_quality_score": statistics.mean([r.get("quality_score", 0.0) for r in type_results]) if type_results else 0
                }
            
            # NUEVO: Análisis por nivel de dificultad
            difficulty_analysis = {}
            difficulties = set()
            for result in valid_results:
                difficulty = result.get("dataset_info", {}).get("difficulty_level")
                if difficulty:
                    difficulties.add(difficulty)
            
            for difficulty in difficulties:
                diff_results = [r for r in valid_results 
                               if r.get("dataset_info", {}).get("difficulty_level") == difficulty]
                diff_successes = [r for r in diff_results if r.get("combined_success", False)]
                
                difficulty_analysis[difficulty] = {
                    "total_samples": len(diff_results),
                    "successful": len(diff_successes),
                    "success_rate": len(diff_successes) / len(diff_results) if diff_results else 0,
                    "avg_quality_score": statistics.mean([r.get("quality_score", 0.0) for r in diff_results]) if diff_results else 0
                }
            
            # NUEVO: Análisis de robustez ante errores
            error_analysis = {
                "with_errors": {
                    "total_samples": 0,
                    "successful": 0,
                    "success_rate": 0,
                    "avg_quality_score": 0,
                    "error_types_breakdown": {}
                },
                "without_errors": {
                    "total_samples": 0,
                    "successful": 0,
                    "success_rate": 0,
                    "avg_quality_score": 0
                }
            }
            
            # Resultados con errores
            error_results = [r for r in valid_results 
                           if r.get("dataset_info", {}).get("has_errors", False)]
            error_successes = [r for r in error_results if r.get("combined_success", False)]
            
            if error_results:
                error_analysis["with_errors"] = {
                    "total_samples": len(error_results),
                    "successful": len(error_successes),
                    "success_rate": len(error_successes) / len(error_results),
                    "avg_quality_score": statistics.mean([r.get("quality_score", 0.0) for r in error_results])
                }
                
                # Análisis por tipo de error
                error_types_breakdown = {}
                for result in error_results:
                    error_types = result.get("dataset_info", {}).get("error_types", [])
                    for error_type in error_types:
                        if error_type not in error_types_breakdown:
                            error_types_breakdown[error_type] = {
                                "total_samples": 0,
                                "successful": 0,
                                "success_rate": 0
                            }
                        
                        error_types_breakdown[error_type]["total_samples"] += 1
                        if result.get("combined_success", False):
                            error_types_breakdown[error_type]["successful"] += 1
                
                # Calcular success rates por tipo de error
                for error_type in error_types_breakdown:
                    total = error_types_breakdown[error_type]["total_samples"]
                    successful = error_types_breakdown[error_type]["successful"]
                    error_types_breakdown[error_type]["success_rate"] = successful / total if total > 0 else 0
                
                error_analysis["with_errors"]["error_types_breakdown"] = error_types_breakdown
            
            # Resultados sin errores
            clean_results = [r for r in valid_results 
                           if not r.get("dataset_info", {}).get("has_errors", False)]
            clean_successes = [r for r in clean_results if r.get("combined_success", False)]
            
            if clean_results:
                error_analysis["without_errors"] = {
                    "total_samples": len(clean_results),
                    "successful": len(clean_successes),
                    "success_rate": len(clean_successes) / len(clean_results),
                    "avg_quality_score": statistics.mean([r.get("quality_score", 0.0) for r in clean_results])
                }
            
            agent_summary = {
                "total_samples": len(valid_results),
                "technical_successful_executions": len(technical_successes),
                "technical_success_rate": len(technical_successes) / len(valid_results) if valid_results else 0,
                "average_quality_score": statistics.mean(quality_scores) if quality_scores else 0,
                "found_in_results_count": len(found_results),
                "found_in_results_rate": len(found_results) / len(valid_results) if valid_results else 0,
                "perfect_hits": len(perfect_hits),
                "perfect_rate": len(perfect_hits) / len(valid_results) if valid_results else 0,
                "top_3_hits": len(top_3_hits),
                "top_3_rate": len(top_3_hits) / len(valid_results) if valid_results else 0,
                "top_5_hits": len(top_5_hits),
                "top_5_rate": len(top_5_hits) / len(valid_results) if valid_results else 0,
                "combined_successful_executions": len(combined_successes),
                "combined_success_rate": len(combined_successes) / len(valid_results) if valid_results else 0,
                "success_rate": len(combined_successes) / len(valid_results) if valid_results else 0,
                "execution_times": {
                    "min": min(execution_times) if execution_times else 0,
                    "max": max(execution_times) if execution_times else 0,
                    "mean": statistics.mean(execution_times) if execution_times else 0,
                    "median": statistics.median(execution_times) if execution_times else 0,
                    "total": sum(execution_times) if execution_times else 0
                },
                "position_stats": {
                    "min_position": min(positions) if positions else None,
                    "max_position": max(positions) if positions else None,
                    "avg_position": statistics.mean(positions) if positions else None
                },
                "tier_distribution": tier_counts,
                "failed_executions": len(valid_results) - len(technical_successes),
                
                # NUEVOS: Análisis detallados por características del dataset
                "query_type_analysis": query_type_analysis,
                "difficulty_analysis": difficulty_analysis,
                "error_robustness_analysis": error_analysis
            }
            
            summary["agents_summary"][agent_name] = agent_summary
        
        return summary

    def save_results_local(self, results: Dict[str, Any]) -> str:
        """Guardar resultados de evaluación local en archivo JSON"""
        
        # Crear directorio de resultados
        results_dir = Path("evaluation/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de archivo con timestamp y información de configuración
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Obtener información de modelo y muestras
        num_samples = results.get("evaluation_metadata", {}).get("dataset_size", "unknown")
        normal_model = os.getenv("OLLAMA_MODEL", "unknown").replace(":", "_").replace("-", "_")
        thinking_model = os.getenv("OLLAMA_MODEL_THINKING", "unknown").replace(":", "_").replace("-", "_")
        
        filename = f"results_local_{num_samples}samples_{normal_model}_{thinking_model}_{timestamp}.json"
        filepath = results_dir / filename
        
        # Guardar resultados
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=numpy_json_serializer)
            
            print(f"💾 Resultados LOCAL guardados en: {filepath}")
            return str(filepath)
            
        except Exception as e:
            print(f"❌ Error guardando resultados: {e}")
            return ""

    def _get_actual_samples_count(self, results: Dict[str, Any]) -> str:
        """Obtener el número real de samples evaluados"""
        if not results:
            return "0"
        
        # Intentar obtener desde metadata primero
        if "evaluation_metadata" in results:
            dataset_size = results["evaluation_metadata"].get("dataset_size", 0)
            if dataset_size > 0:
                return str(dataset_size)
        
        # Fallback: contar resultados únicos por agente
        if "results_by_agent" not in results:
            return "0"
        
        # Obtener el número de resultados del primer agente como aproximación
        agent_results = list(results["results_by_agent"].values())
        if agent_results and len(agent_results[0]) > 0:
            return str(len(agent_results[0]))
        
        return "0"


# Función de conveniencia para usar el sistema local
async def run_evaluation_local(agents: List[str] = None, 
                             max_samples: int = 100, 
                             max_concurrent: int = 3,
                             dataset_path: str = "data/datasets/dataset_generado_final.json") -> str:
    """
    Función de conveniencia para ejecutar evaluación local
    
    Args:
        agents: Lista de agentes a evaluar (None para todos)
        max_samples: Número máximo de muestras a evaluar
        max_concurrent: Número máximo de ejecuciones concurrentes
        dataset_path: Ruta al dataset
    
    Returns:
        Ruta del archivo de resultados generado
    """
    
    # Crear sistema de evaluación local
    evaluator = AgentEvaluationSystemLocal(dataset_path=dataset_path)
    
    # Ejecutar evaluación
    results = await evaluator.evaluate_all_agents_local(
        max_samples=max_samples,
        max_concurrent=max_concurrent,
        agents_to_evaluate=agents
    )
    
    # Guardar resultados
    output_path = evaluator.save_results_local(results)
    
    return output_path 
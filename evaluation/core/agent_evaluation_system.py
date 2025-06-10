"""
Sistema de Evaluación de Agentes PyCiudad con Ground Truth

Sistema comprehensivo que evalúa tanto la robustez técnica como la calidad
de los resultados comparándolos con ground truth del dataset.
"""

import asyncio
import os
import statistics
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json
import numpy as np

# LangGraph imports
from langgraph_sdk import get_client

# Ground truth evaluation
from evaluation.core.ground_truth_evaluator import GroundTruthEvaluator, evaluate_single_execution

# Token tracking oficial de LangChain
# from langchain_core.callbacks import UsageMetadataCallbackHandler


def numpy_json_serializer(obj):
    """Serializador personalizado para manejar tipos numpy"""
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


class AgentEvaluationSystem:
    """Sistema de evaluación para todos los agentes de PyCiudad"""
    
    def __init__(self, 
                 langgraph_url: str = "http://127.0.0.1:2024",
                 dataset_path: str = "data/datasets/dataset_generado_final.json"):
        """
        Inicializar el sistema de evaluación
        
        Args:
            langgraph_url: URL del servidor LangGraph
            dataset_path: Ruta al dataset con ground truth
        """
        self.langgraph_url = langgraph_url
        self.dataset_path = dataset_path
        self.client = None
        self.agents = ["agent_base", "agent_intention", "agent_validation", "agent_ensemble"]
        
        # Capturar configuración de modelos al momento de la evaluación
        self.model_config = self._capture_model_config()
        
        # Inicializar evaluador de ground truth
        self.ground_truth_evaluator = GroundTruthEvaluator()
        
        print(f"🎯 Sistema de Evaluación inicializado")
        print(f"📊 Dataset: {dataset_path}")
        print(f"🤖 Agentes: {', '.join(self.agents)}")
        print(f"🔧 URL LangGraph: {langgraph_url}")
        print(f"📈 Evaluación con Ground Truth: ACTIVADA")
    
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

    async def initialize_client(self):
        """Inicializar cliente de LangGraph"""
        try:
            self.client = get_client(url=self.langgraph_url)
            print("✅ Cliente LangGraph inicializado")
        except Exception as e:
            print(f"❌ Error inicializando cliente: {e}")
            raise

    def load_dataset(self, max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
        """Cargar dataset con ground truth"""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            
            if max_samples:
                dataset = dataset[:max_samples]
                
            print(f"📊 Dataset cargado: {len(dataset)} muestras")
            
            # Verificar que todas las muestras tienen ground truth
            missing_gt = [i for i, sample in enumerate(dataset) if not sample.get("ground_truth_id")]
            if missing_gt:
                print(f"⚠️  {len(missing_gt)} muestras sin ground_truth_id")
            
            return dataset
            
        except Exception as e:
            print(f"❌ Error cargando dataset: {e}")
            return []

    async def create_thread_for_agent(self, agent_name: str) -> str:
        """Crear thread para un agente específico"""
        try:
            # Crear assistant para el agente
            assistant = await self.client.assistants.create(
                graph_id=agent_name,
                config={}
            )
            
            # Crear thread
            thread = await self.client.threads.create()
            
            return thread['thread_id'], assistant['assistant_id']
            
        except Exception as e:
            print(f"❌ Error creando thread para {agent_name}: {e}")
            raise

    async def execute_agent_on_sample(self, agent_config: Dict[str, Any], sample: Dict[str, Any], 
                                      sample_index: int, evaluation_id: str) -> Dict[str, Any]:
        """Ejecutar un agente en una muestra específica"""
        
        start_time = time.time()
        agent_name = agent_config.get('name', 'unknown')
        
        # TODO: Token tracking para LangGraph SDK no está disponible actualmente
        # El sistema de callbacks de LangChain no es compatible con LangGraph SDK
        # Cuando esté disponible, descomentar las siguientes líneas:
        # 
        # from langchain_core.callbacks import UsageMetadataCallbackHandler
        # token_callback = UsageMetadataCallbackHandler()
        
        # Inicializar variables por defecto
        thread_state = None
        run_status = "failed"
        agent_output = {}
        run_id = None
        
        try:
            print(f"🤖 Ejecutando {agent_name} en muestra {sample_index + 1}...")
            
            # Crear thread para este agente si no existe
            thread_id, assistant_id = await self.create_thread_for_agent(agent_name)
            
            # Preparar input para el agente
            agent_input = {
                'user_query': sample.get('user_query', ''),
                'context_from_meta_evaluator': None
            }
            
            # Configuración para metadata (sin callbacks por ahora)
            metadata_config = {
                "model_config": self.model_config,
                "sample_index": sample_index,
                "evaluation_id": evaluation_id
            }
            
            # Ejecutar el agente (sin token tracking por ahora)
            final_run = await self.client.runs.create(
                thread_id=thread_id,
                assistant_id=assistant_id,
                input=agent_input
                # config=run_config  # Comentado hasta que el token tracking esté disponible
            )
            
            # Esperar a que termine la ejecución
            run_id = final_run.run_id if hasattr(final_run, 'run_id') else final_run['run_id']
            await self.client.runs.join(thread_id, run_id)
            
            # Obtener estado final del thread
            thread_state = await self.client.threads.get_state(thread_id)
            run_status = "completed"
            
        except Exception as e:
            print(f"❌ Error ejecutando {agent_name}: {str(e)}")
            # Las variables ya están inicializadas arriba
            
        execution_time = time.time() - start_time
        
        # ========== NUEVA EVALUACIÓN DE CALIDAD ==========
        
        # Extraer output del agente - CORRECCIÓN FINAL
        if thread_state:
            try:
                # El thread_state es un dict con una clave 'values' que contiene el estado real
                if isinstance(thread_state, dict) and 'values' in thread_state:
                    # Acceder directamente a la clave 'values'
                    agent_output = thread_state['values']
                elif hasattr(thread_state, 'values') and callable(thread_state.values):
                    # Si es un dict, values() devuelve dict_values, necesitamos el primer elemento
                    values_result = list(thread_state.values())
                    agent_output = values_result[0] if values_result else {}
                else:
                    agent_output = {}
                    print(f"⚠️ Estructura de thread_state no reconocida: {type(thread_state)}")
            except Exception as e:
                agent_output = {}
                print(f"⚠️ Error extrayendo agent_output: {e}")
        else:
            agent_output = {}
        
        # Debug: imprimir información sobre el agent_output para diagnosticar
        if isinstance(agent_output, dict):
            print(f"🔍 Agent output keys: {list(agent_output.keys())}")
            if 'final_candidates' in agent_output:
                print(f"🔍 final_candidates found: {len(agent_output.get('final_candidates', []))} items")
                # Mostrar los primeros candidatos para verificar
                candidates = agent_output.get('final_candidates', [])[:3]
                for i, candidate in enumerate(candidates):
                    if isinstance(candidate, dict) and 'id' in candidate:
                        print(f"🔍 Candidate {i+1}: ID={candidate['id']}")
        else:
            print(f"⚠️ Agent output no es dict: {type(agent_output)}")
        
        # Evaluar calidad contra ground truth
        quality_evaluation = evaluate_single_execution(agent_output, sample)
        
        # Determinar éxito combinado (técnico + calidad)
        technical_success = run_status not in ["error", "failed"]
        quality_score = quality_evaluation.get("quality_score", 0.0)
        
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
        ground_truth_id = sample.get("ground_truth_id")
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
        
        # Metadatos de tokens: Por ahora usar valores por defecto
        # TODO: Implementar tracking real cuando esté disponible en LangGraph SDK
        # LangGraph SDK actual no soporta callbacks de LangChain para token tracking
        token_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'total_tokens': 0,
            'detailed_usage': {},
            'note': 'Token tracking no disponible con LangGraph SDK actual'
        }
        
        # Calcular costo estimado (por ahora $0.00)
        estimated_cost = 0.0
        
        # Construir resultado completo con información rica
        result = {
            "agent_name": agent_name,
            "sample_id": sample.get("id"),
            "input_query": agent_input["user_query"],
            "execution_time_seconds": execution_time,
            
            # Métricas técnicas (antiguas)
            "run_status": run_status,
            "technical_success": technical_success,
            
            # Información rica del dataset (NUEVA)
            "dataset_info": dataset_info,
            
            # Información detallada de ground truth (MEJORADA)
            "ground_truth_info": detailed_ground_truth,
            
            # Métricas de calidad
            "quality_score": quality_score,
            "position_found": quality_evaluation.get("position_found"),
            "total_candidates": quality_evaluation.get("total_candidates", 0),
            "found_in_results": quality_evaluation.get("found_in_results", False),
            "scoring_tier": quality_evaluation.get("scoring_tier", "not_found"),
            
            # Métricas combinadas
            "combined_success": technical_success and quality_score > 0.0,
            "success": technical_success and quality_score > 0.0,  # Nueva definición de éxito
            
            # Metadata y contexto
            "run_id": run_id,
            "thread_id": thread_id,
            "timestamp": datetime.now().isoformat(),
            "agent_output": agent_output,
            "metadata": metadata_config,
            "original_sample": sample,
            "ground_truth_evaluation": quality_evaluation,
            
            # Métricas de uso de tokens
            "token_usage": token_usage,
            "estimated_cost": estimated_cost
        }
        
        # Log detallado del resultado con información rica
        status_icon = "✅" if result["success"] else "❌"
        tech_icon = "✓" if technical_success else "✗"
        query_type = dataset_info.get("query_type", "unknown")
        difficulty = dataset_info.get("difficulty_level", "unknown")
        has_errors = "🔴" if dataset_info.get("has_errors") else "🟢"
        
        print(f"{status_icon} {agent_name} - {execution_time:.2f}s - "
              f"Tech: {tech_icon} - "
              f"Quality: {quality_score:.2f} - "
              f"Position: {quality_evaluation.get('position_found', 'N/A')} - "
              f"Type: {query_type} - Difficulty: {difficulty} - Errors: {has_errors} - "
              f"Sample: {sample.get('id')}")
        
        return result

    async def evaluate_agent_on_dataset(self, 
                                       agent_name: str, 
                                       dataset: List[Dict[str, Any]],
                                       max_concurrent: int = 3) -> List[Dict[str, Any]]:
        """
        Evaluar un agente específico en todo el dataset
        
        Args:
            agent_name: Nombre del agente a evaluar
            dataset: Dataset de muestras con ground truth
            max_concurrent: Número máximo de ejecuciones concurrentes
        
        Returns:
            Lista de resultados de todas las ejecuciones
        """
        print(f"\n🤖 Evaluando agente: {agent_name}")
        print(f"📊 Muestras a procesar: {len(dataset)}")
        
        results = []
        
        # Crear un thread para este agente
        thread_id, assistant_id = await self.create_thread_for_agent(agent_name)
        
        # Procesar muestras en lotes para evitar saturar el servicio
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_sample(sample_index, sample):
            async with semaphore:
                return await self.execute_agent_on_sample(
                    {"name": agent_name}, sample, sample_index, f"eval_{agent_name}"
                )
        
        # Ejecutar todas las muestras
        tasks = [process_sample(i, sample) for i, sample in enumerate(dataset)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filtrar excepciones y convertir a resultados válidos
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"❌ Excepción en muestra {i}: {result}")
                valid_results.append({
                    "agent_name": agent_name,
                    "sample_id": dataset[i].get("id"),
                    "error": str(result),
                    "timestamp": datetime.now().isoformat(),
                    "success": False,
                    "technical_success": False,
                    "quality_score": 0.0,
                    "combined_success": False
                })
            else:
                valid_results.append(result)
        
        print(f"✅ {agent_name} completado: {len(valid_results)} resultados")
        return valid_results

    async def evaluate_all_agents(self, 
                                 max_samples: Optional[int] = 100,
                                 max_concurrent: int = 3) -> Dict[str, Any]:
        """
        Evaluar todos los agentes en el dataset
        
        Args:
            max_samples: Número máximo de muestras del dataset (None para todo)
            max_concurrent: Número máximo de ejecuciones concurrentes por agente
        
        Returns:
            Resultados completos de la evaluación con métricas de calidad
        """
        print("🚀 Iniciando evaluación completa de agentes CON GROUND TRUTH")
        
        # Cargar dataset
        dataset = self.load_dataset(max_samples)
        
        # Inicializar cliente
        await self.initialize_client()
        
        # Evaluar cada agente
        all_results = {}
        total_start_time = time.time()
        
        for agent_name in self.agents:
            try:
                agent_results = await self.evaluate_agent_on_dataset(
                    agent_name, dataset, max_concurrent
                )
                all_results[agent_name] = agent_results
                
            except Exception as e:
                print(f"❌ Error evaluando {agent_name}: {e}")
                all_results[agent_name] = []
        
        total_execution_time = time.time() - total_start_time
        
        # Compilar estadísticas (ACTUALIZADO para incluir métricas de calidad)
        evaluation_summary = self.compile_evaluation_statistics(all_results, total_execution_time)
        
        # Resultado final
        final_results = {
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_execution_time_seconds": total_execution_time,
                "dataset_size": len(dataset),
                "agents_evaluated": list(all_results.keys()),
                "langgraph_url": self.langgraph_url,
                "model_config": self.model_config,
                "evaluation_type": "ground_truth_based",
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
        
        return final_results
    
    def compile_evaluation_statistics(self, 
                                    all_results: Dict[str, List[Dict[str, Any]]],
                                    total_time: float) -> Dict[str, Any]:
        """Compilar estadísticas de la evaluación incluyendo métricas de calidad"""
        
        stats = {
            "total_execution_time_seconds": total_time,
            "agents_summary": {}
        }
        
        for agent_name, results in all_results.items():
            if not results:
                continue
                
            # Separar por tipo de éxito
            technical_success_results = [r for r in results if r.get("technical_success", False)]
            combined_success_results = [r for r in results if r.get("combined_success", False)]
            error_results = [r for r in results if not r.get("technical_success", False)]
            
            # Calcular estadísticas de tiempo
            execution_times = [r.get("execution_time_seconds", 0) for r in results]
            
            # Calcular estadísticas de calidad
            quality_scores = [r.get("quality_score", 0.0) for r in results]
            positions_found = [r.get("position_found") for r in results if r.get("position_found")]
            
            # Contar por tiers de scoring
            tier_counts = {}
            for result in results:
                tier = result.get("scoring_tier", "error")
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            # NUEVO: Análisis por tipo de consulta
            query_type_analysis = {}
            query_types = set()
            for result in results:
                query_type = result.get("dataset_info", {}).get("query_type")
                if query_type:
                    query_types.add(query_type)
            
            for query_type in query_types:
                type_results = [r for r in results 
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
            for result in results:
                difficulty = result.get("dataset_info", {}).get("difficulty_level")
                if difficulty:
                    difficulties.add(difficulty)
            
            for difficulty in difficulties:
                diff_results = [r for r in results 
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
            error_results_data = [r for r in results 
                           if r.get("dataset_info", {}).get("has_errors", False)]
            error_successes = [r for r in error_results_data if r.get("combined_success", False)]
            
            if error_results_data:
                error_analysis["with_errors"] = {
                    "total_samples": len(error_results_data),
                    "successful": len(error_successes),
                    "success_rate": len(error_successes) / len(error_results_data),
                    "avg_quality_score": statistics.mean([r.get("quality_score", 0.0) for r in error_results_data])
                }
                
                # Análisis por tipo de error
                error_types_breakdown = {}
                for result in error_results_data:
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
            clean_results = [r for r in results 
                           if not r.get("dataset_info", {}).get("has_errors", False)]
            clean_successes = [r for r in clean_results if r.get("combined_success", False)]
            
            if clean_results:
                error_analysis["without_errors"] = {
                    "total_samples": len(clean_results),
                    "successful": len(clean_successes),
                    "success_rate": len(clean_successes) / len(clean_results),
                    "avg_quality_score": statistics.mean([r.get("quality_score", 0.0) for r in clean_results])
                }
            
            agent_stats = {
                # Métricas básicas
                "total_samples": len(results),
                
                # Métricas técnicas (antiguas)
                "technical_successful_executions": len(technical_success_results),
                "technical_success_rate": len(technical_success_results) / len(results) if results else 0,
                
                # Métricas de calidad
                "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
                "found_in_results_count": len([r for r in results if r.get("found_in_results", False)]),
                "found_in_results_rate": len([r for r in results if r.get("found_in_results", False)]) / len(results) if results else 0,
                
                # Métricas por posición
                "perfect_hits": tier_counts.get("perfect", 0),
                "perfect_rate": tier_counts.get("perfect", 0) / len(results) if results else 0,
                "top_3_hits": tier_counts.get("perfect", 0) + tier_counts.get("top_3", 0),
                "top_3_rate": (tier_counts.get("perfect", 0) + tier_counts.get("top_3", 0)) / len(results) if results else 0,
                "top_5_hits": tier_counts.get("perfect", 0) + tier_counts.get("top_3", 0) + tier_counts.get("top_5", 0),
                "top_5_rate": (tier_counts.get("perfect", 0) + tier_counts.get("top_3", 0) + tier_counts.get("top_5", 0)) / len(results) if results else 0,
                
                # Métricas combinadas
                "combined_successful_executions": len(combined_success_results),
                "combined_success_rate": len(combined_success_results) / len(results) if results else 0,
                "success_rate": len(combined_success_results) / len(results) if results else 0,  # Nueva definición
                
                # Estadísticas de tiempo
                "execution_times": {
                    "min": min(execution_times) if execution_times else 0,
                    "max": max(execution_times) if execution_times else 0,
                    "mean": statistics.mean(execution_times) if execution_times else 0,
                    "median": statistics.median(execution_times) if execution_times else 0,
                    "total": sum(execution_times)
                },
                
                # Estadísticas de posición
                "position_stats": {
                    "min_position": min(positions_found) if positions_found else None,
                    "max_position": max(positions_found) if positions_found else None,
                    "avg_position": sum(positions_found) / len(positions_found) if positions_found else None
                },
                
                # Distribución por tiers
                "tier_distribution": tier_counts,
                
                # Errores
                "failed_executions": len(error_results),
                
                # NUEVOS: Análisis detallados por características del dataset
                "query_type_analysis": query_type_analysis,
                "difficulty_analysis": difficulty_analysis,
                "error_robustness_analysis": error_analysis
            }
            
            stats["agents_summary"][agent_name] = agent_stats
        
        return stats
    
    def save_results(self, results: Dict[str, Any], output_path: str = None):
        """Guardar resultados de la evaluación"""
        if output_path is None:
            # Generar nombre descriptivo
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Obtener información para el nombre
            num_samples = self._get_actual_samples_count(results)
            normal_model = self._get_clean_model_name("OLLAMA_MODEL")
            thinking_model = self._get_clean_model_name("OLLAMA_MODEL_THINKING")
            
            # Construir nombre descriptivo
            filename = f"results_{num_samples}samples_{normal_model}_{thinking_model}_{timestamp}.json"
            output_path = f"evaluation/results/{filename}"
        
        # Crear directorio si no existe
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=numpy_json_serializer)
        
        print(f"💾 Resultados guardados en: {output_path}")
        return output_path

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

    def _get_clean_model_name(self, env_var: str) -> str:
        """Obtener nombre limpio del modelo desde variable de entorno"""
        model_name = os.getenv(env_var)
        if not model_name:
            return "unknown"
        
        # Limpiar el nombre del modelo (remover caracteres especiales, espacios, etc.)
        # Convertir a minúsculas y reemplazar caracteres problemáticos
        clean_name = model_name.lower()
        clean_name = clean_name.replace(":", "_")
        clean_name = clean_name.replace("-", "_")
        clean_name = clean_name.replace(" ", "_")
        clean_name = clean_name.replace(".", "_")
        
        return clean_name

    def _calculate_estimated_cost(self, total_tokens: int) -> float:
        # Implementa la lógica para calcular el costo estimado basado en el número total de tokens
        # Este es un ejemplo básico y debería ser reemplazado por una implementación real basada en precios
        return total_tokens * 0.0001  # Ejemplo: 1 token = $0.0001


async def main():
    """Función principal para ejecutar la evaluación"""
    
    # Configuración
    evaluator = AgentEvaluationSystem()
    
    # Ejecutar evaluación
    # Empezamos con pocas muestras para prueba
    results = await evaluator.evaluate_all_agents(max_samples=10, max_concurrent=2)
    
    # Guardar resultados
    output_path = evaluator.save_results(results)
    
    # Mostrar resumen
    print("\n" + "="*80)
    print("📊 RESUMEN DE EVALUACIÓN")
    print("="*80)
    
    summary = results["evaluation_summary"]
    print(f"⏱️  Tiempo total: {summary['total_execution_time_seconds']:.2f} segundos")
    
    for agent, stats in summary["agents_summary"].items():
        print(f"\n🤖 {agent.upper()}:")
        print(f"   ✅ Éxito: {stats['successful_executions']}/{stats['total_samples']} ({stats['success_rate']*100:.1f}%)")
        print(f"   ⏱️  Tiempo promedio: {stats['execution_times']['mean']:.2f}s")
        print(f"   📊 Rango: {stats['execution_times']['min']:.2f}s - {stats['execution_times']['max']:.2f}s")
    
    print(f"\n💾 Resultados completos en: {output_path}")


if __name__ == "__main__":
    asyncio.run(main()) 
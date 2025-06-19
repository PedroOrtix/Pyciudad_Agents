"""
Utilidad para fusionar múltiples JSONs de resultados de evaluación.

Esta utilidad permite combinar varios archivos JSON de resultados que comparten
el mismo dataset y configuración de modelos, pero evalúan diferentes agentes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Union, Optional
from datetime import datetime
import copy

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class JSONMerger:
    """
    Clase para fusionar múltiples JSONs de resultados de evaluación.
    """
    
    def __init__(self):
        self.merged_data = None
        self.source_files = []
    
    def merge_evaluation_jsons(self, json_files: List[Union[str, Path]], 
                              output_file: Optional[Union[str, Path]] = None,
                              validate_compatibility: bool = True) -> Dict[str, Any]:
        """
        Fusiona múltiples JSONs de resultados de evaluación.
        
        Args:
            json_files: Lista de rutas a los archivos JSON a fusionar
            output_file: Ruta donde guardar el JSON fusionado (opcional)
            validate_compatibility: Si validar que los JSONs sean compatibles
            
        Returns:
            Dict con los datos fusionados
            
        Raises:
            ValueError: Si los JSONs no son compatibles
            FileNotFoundError: Si algún archivo no existe
        """
        if not json_files:
            raise ValueError("Se debe proporcionar al menos un archivo JSON")
        
        logger.info(f"Iniciando fusión de {len(json_files)} archivos JSON")
        
        # Cargar todos los JSONs
        loaded_jsons = []
        for file_path in json_files:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                loaded_jsons.append((path, data))
                logger.info(f"Cargado: {path.name}")
        
        # Validar compatibilidad si se solicita
        if validate_compatibility:
            self._validate_compatibility(loaded_jsons)
        
        # Fusionar los datos
        merged = self._merge_data(loaded_jsons)
        
        # Guardar si se especifica archivo de salida
        if output_file:
            self._save_merged_data(merged, output_file)
        
        self.merged_data = merged
        self.source_files = [str(path) for path, _ in loaded_jsons]
        
        logger.info("Fusión completada exitosamente")
        return merged
    
    def _validate_compatibility(self, loaded_jsons: List[tuple]) -> None:
        """
        Valida que los JSONs sean compatibles para fusión.
        """
        if len(loaded_jsons) < 2:
            return
        
        base_path, base_data = loaded_jsons[0]
        base_metadata = base_data.get('evaluation_metadata', {})
        
        # Campos que deben ser idénticos
        critical_fields = [
            'dataset_size', 
            'model_config', 
            'evaluation_type', 
            'scoring_system'
        ]
        
        for path, data in loaded_jsons[1:]:
            metadata = data.get('evaluation_metadata', {})
            
            for field in critical_fields:
                if field in base_metadata and field in metadata:
                    if base_metadata[field] != metadata[field]:
                        logger.warning(
                            f"Diferencia en {field} entre {base_path.name} y {path.name}: "
                            f"{base_metadata[field]} vs {metadata[field]}"
                        )
        
        logger.info("Validación de compatibilidad completada")
    
    def _merge_data(self, loaded_jsons: List[tuple]) -> Dict[str, Any]:
        """
        Fusiona los datos de los JSONs cargados.
        """
        # Usar el primer JSON como base
        base_path, base_data = loaded_jsons[0]
        merged = copy.deepcopy(base_data)
        
        # Recopilar información de fusión
        all_agents = set()
        total_execution_time = 0
        all_source_files = []
        
        # Procesar todos los JSONs
        for path, data in loaded_jsons:
            all_source_files.append(str(path))
            
            # Agregar agentes a la lista
            if 'evaluation_metadata' in data and 'agents_evaluated' in data['evaluation_metadata']:
                all_agents.update(data['evaluation_metadata']['agents_evaluated'])
            
            # Sumar tiempo de ejecución
            if 'evaluation_metadata' in data:
                exec_time = data['evaluation_metadata'].get('total_execution_time_seconds', 0)
                total_execution_time += exec_time
            
            # Fusionar results_by_agent
            if 'results_by_agent' in data:
                if 'results_by_agent' not in merged:
                    merged['results_by_agent'] = {}
                
                for agent_name, agent_results in data['results_by_agent'].items():
                    if agent_name in merged['results_by_agent']:
                        logger.warning(f"Agente duplicado encontrado: {agent_name}")
                        # Decidir estrategia: sobrescribir, fusionar o ignorar
                        merged['results_by_agent'][agent_name].extend(agent_results)
                    else:
                        merged['results_by_agent'][agent_name] = agent_results
            
            # Fusionar agents_summary en evaluation_summary
            if 'evaluation_summary' in data and 'agents_summary' in data['evaluation_summary']:
                if 'evaluation_summary' not in merged:
                    merged['evaluation_summary'] = {}
                if 'agents_summary' not in merged['evaluation_summary']:
                    merged['evaluation_summary']['agents_summary'] = {}
                
                for agent_name, agent_summary in data['evaluation_summary']['agents_summary'].items():
                    merged['evaluation_summary']['agents_summary'][agent_name] = agent_summary
        
        # Actualizar metadata del JSON fusionado
        self._update_merged_metadata(merged, all_agents, total_execution_time, all_source_files)
        
        # Recalcular estadísticas generales
        self._recalculate_general_statistics(merged)
        
        return merged
    
    def _update_merged_metadata(self, merged: Dict[str, Any], 
                               all_agents: set, 
                               total_execution_time: float,
                               source_files: List[str]) -> None:
        """
        Actualiza la metadata del JSON fusionado.
        """
        if 'evaluation_metadata' not in merged:
            merged['evaluation_metadata'] = {}
        
        # Actualizar información de fusión
        merged['evaluation_metadata']['merged_from_files'] = source_files
        merged['evaluation_metadata']['merge_timestamp'] = datetime.now().isoformat()
        merged['evaluation_metadata']['agents_evaluated'] = sorted(list(all_agents))
        merged['evaluation_metadata']['total_execution_time_seconds'] = total_execution_time
        merged['evaluation_metadata']['is_merged'] = True
        merged['evaluation_metadata']['merge_file_count'] = len(source_files)
    
    def _recalculate_general_statistics(self, merged: Dict[str, Any]) -> None:
        """
        Recalcula las estadísticas generales basándose en los datos fusionados.
        """
        if 'evaluation_summary' not in merged or 'agents_summary' not in merged['evaluation_summary']:
            return
        
        agents_summaries = merged['evaluation_summary']['agents_summary']
        
        # Calcular estadísticas agregadas
        total_executions = sum(agent.get('total_samples', 0) for agent in agents_summaries.values())
        total_agents = len(agents_summaries)
        
        # Promedios ponderados
        if total_executions > 0:
            weighted_technical_success = sum(
                agent.get('technical_success_rate', 0) * agent.get('total_samples', 0)
                for agent in agents_summaries.values()
            ) / total_executions
            
            weighted_combined_success = sum(
                agent.get('combined_success_rate', 0) * agent.get('total_samples', 0)
                for agent in agents_summaries.values()
            ) / total_executions
            
            weighted_quality_score = sum(
                agent.get('average_quality_score', 0) * agent.get('total_samples', 0)
                for agent in agents_summaries.values()
            ) / total_executions
        else:
            weighted_technical_success = 0
            weighted_combined_success = 0
            weighted_quality_score = 0
        
        # Actualizar o crear general_statistics
        if 'general_statistics' not in merged:
            merged['general_statistics'] = {}
        
        merged['general_statistics'].update({
            'total_executions': total_executions,
            'total_agents': total_agents,
            'technical_success_rate': weighted_technical_success,
            'combined_success_rate': weighted_combined_success,
            'overall_success_rate': weighted_combined_success,
            'average_quality_score': weighted_quality_score,
            'is_merged_calculation': True
        })
        
        # Actualizar execution_summary si existe
        if 'evaluation_summary' in merged:
            merged['evaluation_summary']['total_execution_time_seconds'] = \
                merged['evaluation_metadata'].get('total_execution_time_seconds', 0)
    
    def _save_merged_data(self, merged_data: Dict[str, Any], output_file: Union[str, Path]) -> None:
        """
        Guarda los datos fusionados en un archivo JSON.
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Datos fusionados guardados en: {output_path}")
    
    def get_merge_summary(self) -> Dict[str, Any]:
        """
        Retorna un resumen de la operación de fusión.
        """
        if not self.merged_data:
            return {"error": "No se ha realizado ninguna fusión"}
        
        return {
            "source_files": self.source_files,
            "merged_agents": self.merged_data.get('evaluation_metadata', {}).get('agents_evaluated', []),
            "total_executions": self.merged_data.get('general_statistics', {}).get('total_executions', 0),
            "merge_timestamp": self.merged_data.get('evaluation_metadata', {}).get('merge_timestamp'),
            "is_merged": True
        }


def merge_evaluation_results(json_files: List[Union[str, Path]], 
                           output_file: Optional[Union[str, Path]] = None,
                           validate_compatibility: bool = True) -> Dict[str, Any]:
    """
    Función de conveniencia para fusionar JSONs de evaluación.
    
    Args:
        json_files: Lista de rutas a los archivos JSON
        output_file: Archivo de salida (opcional)
        validate_compatibility: Si validar compatibilidad
    
    Returns:
        Datos fusionados
    """
    merger = JSONMerger()
    return merger.merge_evaluation_jsons(json_files, output_file, validate_compatibility)


# Ejemplo de uso
if __name__ == "__main__":
    # Ejemplo de cómo usar la utilidad
    files_to_merge = [
        "results/results_local_agent_base.json",
        "results/results_local_agent_intention.json", 
        "results/results_local_agent_validation.json"
    ]
    
    output_file = "results/merged_results.json"
    
    try:
        merged_data = merge_evaluation_results(
            json_files=files_to_merge,
            output_file=output_file,
            validate_compatibility=True
        )
        
        print("Fusión completada exitosamente")
        print(f"Total de agentes: {len(merged_data.get('evaluation_metadata', {}).get('agents_evaluated', []))}")
        
    except Exception as e:
        print(f"Error durante la fusión: {e}") 
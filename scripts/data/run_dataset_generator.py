#!/usr/bin/env python3
"""
Script para ejecutar el agente generador de datasets
"""

import os
import sys
from datetime import datetime

# Añadir el directorio raíz al path para importar módulos
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data.Agent_dataset_generator.agent_dataset_generator import app_dataset_generator
from data.Agent_dataset_generator.states_dataset import GraphStateInput

def run_dataset_generator(sample_size=10, output_filename=None):
    """
    Ejecuta el agente generador de datasets
    
    Args:
        sample_size (int): Número de direcciones a procesar (None para todas)
        output_filename (str): Nombre del archivo de salida
    """
    
    # Generar nombre de archivo con timestamp si no se proporciona
    if output_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"dataset_generado_{timestamp}.json"
    
    print("=== AGENTE GENERADOR DE DATASETS ===")
    print(f"Tamaño de muestra: {sample_size if sample_size else 'Todas las direcciones'}")
    print(f"Archivo de salida: {output_filename}")
    print("=" * 50)
    
    try:
        # Configurar estado inicial
        initial_state = GraphStateInput(
            sample_size=sample_size,
            output_filename=output_filename,
            variations_per_address=5
        )
        
        print("Iniciando generación de dataset...")
        
        # Ejecutar el agente (síncrono)
        result = app_dataset_generator.invoke(initial_state)
        
        print("\n=== RESULTADOS ===")
        print(f"Dataset guardado en: {result.get('dataset_path', 'No disponible')}")
        print(f"Total de entradas generadas: {result.get('statistics', {}).get('total_entries', 0)}")
        print(f"Con errores: {result.get('statistics', {}).get('with_errors', 0)}")
        print(f"Sin errores: {result.get('statistics', {}).get('without_errors', 0)}")
        
        print("\nDistribución por dificultad:")
        by_difficulty = result.get('statistics', {}).get('by_difficulty', {})
        for level, count in by_difficulty.items():
            print(f"  - {level}: {count}")
        
        print("\nDistribución por tipo de consulta:")
        by_query_type = result.get('statistics', {}).get('by_query_type', {})
        for query_type, count in by_query_type.items():
            print(f"  - {query_type}: {count}")
        
        return result
        
    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        raise

def main():
    """Función principal para ejecutar desde línea de comandos"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generador de datasets para entrenamiento")
    parser.add_argument("--sample-size", type=int, default=10, 
                       help="Número de direcciones a procesar (por defecto: 10)")
    parser.add_argument("--output", type=str, 
                       help="Nombre del archivo de salida (por defecto: timestamp)")
    parser.add_argument("--all", action="store_true",
                       help="Procesar todas las direcciones del ground truth")
    
    args = parser.parse_args()
    
    sample_size = None if args.all else args.sample_size
    
    # Ejecutar el generador (síncrono)
    result = run_dataset_generator(
        sample_size=sample_size,
        output_filename=args.output
    )
    
    print(f"\n✅ Dataset generado exitosamente: {result.get('dataset_path', 'No disponible')}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
Script de ejemplo para usar la utilidad de fusión de JSONs.

Este script demuestra cómo usar JSONMerger para combinar múltiples
archivos de resultados de evaluación.
"""

import sys
from pathlib import Path

# Agregar el directorio padre al path para importaciones
sys.path.append(str(Path(__file__).parent.parent))

from utils.json_merger import merge_evaluation_results, JSONMerger


def main():
    """
    Ejemplo principal de uso de la utilidad de fusión.
    """
    
    # Definir los archivos a fusionar (ajusta estas rutas a tus archivos reales)
    json_files = [
        "evaluation/results/results_local_1500samples_phi4_14b_phi4_reasoning_14b_20250603_081958.json",
        # Agrega aquí más archivos JSON que quieras fusionar
    ]
    
    # Verificar que los archivos existen
    existing_files = []
    for file_path in json_files:
        path = Path(file_path)
        if path.exists():
            existing_files.append(file_path)
            print(f"✓ Encontrado: {path.name}")
        else:
            print(f"✗ No encontrado: {file_path}")
    
    if len(existing_files) < 2:
        print("\n⚠️  Necesitas al menos 2 archivos JSON para realizar la fusión.")
        print("Ajusta las rutas en este script o añade más archivos de resultados.")
        return
    
    # Definir archivo de salida
    output_file = "evaluation/results/merged_evaluation_results.json"
    
    print(f"\n🔄 Iniciando fusión de {len(existing_files)} archivos...")
    
    try:
        # Opción 1: Usar la función directa
        merged_data = merge_evaluation_results(
            json_files=existing_files,
            output_file=output_file,
            validate_compatibility=True
        )
        
        print("✅ Fusión completada exitosamente!")
        
        # Mostrar resumen
        agents = merged_data.get('evaluation_metadata', {}).get('agents_evaluated', [])
        total_executions = merged_data.get('general_statistics', {}).get('total_executions', 0)
        
        print("📊 Resumen de la fusión:")
        print(f"   • Agentes fusionados: {', '.join(agents)}")
        print(f"   • Total de ejecuciones: {total_executions}")
        print(f"   • Archivo generado: {output_file}")
        
        # Opción 2: Usar la clase directamente (para más control)
        print("\n🔍 Ejemplo usando la clase JSONMerger:")
        
        merger = JSONMerger()
        merger.merge_evaluation_jsons(
                json_files=existing_files,
                validate_compatibility=True
            )
        
        # Obtener resumen detallado
        summary = merger.get_merge_summary()
        print(f"   • Archivos origen: {len(summary['source_files'])}")
        print(f"   • Timestamp fusión: {summary['merge_timestamp']}")
        print(f"   • Agentes: {summary['merged_agents']}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except ValueError as e:
        print(f"❌ Error de validación: {e}")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")


def list_available_json_files():
    """
    Lista los archivos JSON disponibles en la carpeta de resultados.
    """
    results_dir = Path("evaluation/results")
    if not results_dir.exists():
        print("❌ No existe la carpeta 'evaluation/results'")
        return []
    
    json_files = list(results_dir.glob("*.json"))
    
    if not json_files:
        print("❌ No se encontraron archivos JSON en la carpeta 'evaluation/results'")
        return []
    
    print("📁 Archivos JSON encontrados en 'evaluation/results/':")
    for i, file_path in enumerate(json_files, 1):
        print(f"   {i}. {file_path.name}")
    
    return json_files


if __name__ == "__main__":
    print("🔧 Utilidad de Fusión de JSONs de Evaluación")
    print("=" * 50)
    
    # Listar archivos disponibles
    available_files = list_available_json_files()
    
    if available_files:
        print("\n📝 Para usar esta utilidad:")
        print(f"   1. Edita el script y ajusta las rutas en 'json_files'")
        print(f"   2. Ejecuta: python merge_example.py")
        print(f"   3. Revisa el archivo fusionado en 'evaluation/results/merged_evaluation_results.json'")
        
        # Ejecutar ejemplo si hay archivos
        print("\n" + "=" * 50)
        main()
    else:
        print("\n💡 Primero ejecuta algunas evaluaciones para generar archivos JSON que fusionar.") 
#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias consolidadas están instaladas correctamente.

Uso: python scripts/verify_dependencies.py
"""

import sys
import importlib
from typing import List, Tuple, Dict

def check_dependencies() -> Dict[str, List[Tuple[str, bool, str]]]:
    """
    Verifica las dependencias principales del proyecto.
    
    Returns:
        Diccionario con categorías y resultados de verificación
    """
    
    dependencies = {
        "CORE": [
            ("langchain", "LangChain framework"),
            ("langgraph", "LangGraph para agentes"),
            ("langchain_ollama", "LangChain Ollama integration"),
            ("langchain_community", "LangChain community tools"),
            ("pydantic", "Validación de datos"),
        ],
        "EVALUATION": [
            ("langgraph_sdk", "LangGraph SDK"),
        ],
        "ANALYSIS": [
            ("pandas", "Análisis de datos"),
            ("numpy", "Computación numérica"),
            ("scipy", "Estadísticas avanzadas"),
        ],
        "VISUALIZATION": [
            ("matplotlib", "Gráficos básicos"),
            ("seaborn", "Gráficos estadísticos"),
        ],
        "UTILITIES": [
            ("dateutil", "Procesamiento de fechas"),
        ]
    }
    
    results = {}
    
    for category, deps in dependencies.items():
        results[category] = []
        
        for module_name, description in deps:
            try:
                importlib.import_module(module_name)
                results[category].append((module_name, True, description))
            except ImportError as e:
                results[category].append((module_name, False, f"{description} - ERROR: {e}"))
    
    return results

def print_results(results: Dict[str, List[Tuple[str, bool, str]]]) -> bool:
    """
    Imprime los resultados de verificación.
    
    Returns:
        True si todas las dependencias están instaladas
    """
    all_ok = True
    
    print("🔍 VERIFICACIÓN DE DEPENDENCIAS")
    print("=" * 60)
    
    for category, deps in results.items():
        print(f"\n📁 {category}:")
        
        for module_name, success, description in deps:
            if success:
                print(f"  ✅ {module_name:<20} - {description}")
            else:
                print(f"  ❌ {module_name:<20} - {description}")
                all_ok = False
    
    print("\n" + "=" * 60)
    
    if all_ok:
        print("🎉 TODAS las dependencias están instaladas correctamente!")
        print("✅ El proyecto está listo para usar.")
    else:
        print("⚠️  FALTAN algunas dependencias.")
        print("💡 Ejecuta: pip install -r requirements.txt")
    
    return all_ok

def main():
    """Función principal"""
    print("Verificando dependencias del proyecto PyCiudad_Agents...\n")
    
    try:
        results = check_dependencies()
        all_ok = print_results(results)
        
        # Código de salida
        sys.exit(0 if all_ok else 1)
        
    except Exception as e:
        print(f"❌ Error durante la verificación: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
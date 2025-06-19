"""
Módulo Utils - Utilidades y Herramientas

Herramientas auxiliares, utilidades de desarrollo y
funciones de soporte para el sistema de evaluación.
"""

# Importar utilidades disponibles cuando sea necesario
from .json_merger import JSONMerger, merge_evaluation_results

__all__ = ['JSONMerger', 'merge_evaluation_results'] 
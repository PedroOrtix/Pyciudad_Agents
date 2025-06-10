#!/usr/bin/env python3
"""
Script Principal de Evaluación de Agentes

Este script ejecuta el sistema completo de evaluación de agentes,
incluyendo la ejecución contra el dataset y el análisis de métricas.

Uso:
    python run_evaluation.py --samples 50 --concurrent 2
    python run_evaluation.py --full-dataset
    python run_evaluation.py --analyze-only
"""

import argparse
import asyncio
import sys
from pathlib import Path
import json
import numpy as np

# Agregar el directorio raíz al path para importar módulos
sys.path.append(str(Path(__file__).parent.parent.parent))

from evaluation.core.agent_evaluation_system import AgentEvaluationSystem
# from evaluation.analysis.metrics_analyzer import MetricsAnalyzer  # TODO: Implementar


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


def parse_arguments():
    """Configurar argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Sistema de Evaluación de Agentes PyCiudad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Evaluar 50 muestras con 2 ejecuciones concurrentes
  python run_evaluation.py --samples 50 --concurrent 2
  
  # Evaluar todo el dataset (puede tomar mucho tiempo)
  python run_evaluation.py --full-dataset
  
  # Solo analizar resultados existentes
  python run_evaluation.py --analyze-only
  
  # Configurar URL personalizada de LangGraph
  python run_evaluation.py --url http://localhost:2024 --samples 20
        """
    )
    
    # Argumentos principales
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Número máximo de muestras del dataset a evaluar (default: 100)"
    )
    
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Evaluar todo el dataset (ignora --samples)"
    )
    
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Número máximo de ejecuciones concurrentes por agente (default: 3)"
    )
    
    parser.add_argument(
        "--url",
        type=str,
        default="http://127.0.0.1:2024",
        help="URL del servicio LangGraph (default: http://127.0.0.1:2024)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/dataset_generado_final.json",
        help="Ruta al archivo del dataset (default: data/datasets/dataset_generado_final.json)"
    )
    
    # Argumentos de control
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Solo analizar resultados existentes, no ejecutar evaluación"
    )
    
    parser.add_argument(
        "--agents",
        nargs="+",
        default=["agent_base", "agent_intention", "agent_validation", "agent_ensemble"],
        help="Lista de agentes a evaluar (default: todos)"
    )
    
    # Argumentos de salida
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation",
        help="Directorio base para guardar resultados (default: evaluation)"
    )
    
    return parser.parse_args()


async def run_evaluation(args):
    """Ejecutar evaluación de agentes"""
    
    print("🚀 Iniciando evaluación de agentes...")
    print(f"📊 Configuración:")
    print(f"   - Muestras: {'Todo el dataset' if args.full_dataset else args.samples}")
    print(f"   - Concurrencia: {args.concurrent}")
    print(f"   - URL LangGraph: {args.url}")
    print(f"   - Agentes: {', '.join(args.agents)}")
    
    # Verificar que el dataset existe
    if not Path(args.dataset).exists():
        print(f"❌ Dataset no encontrado: {args.dataset}")
        return False
    
    # Crear evaluador
    evaluator = AgentEvaluationSystem(
        langgraph_url=args.url,
        dataset_path=args.dataset
    )
    
    # Configurar agentes a evaluar
    evaluator.agents = args.agents
    
    try:
        # Ejecutar evaluación
        max_samples = None if args.full_dataset else args.samples
        results = await evaluator.evaluate_all_agents(
            max_samples=max_samples,
            max_concurrent=args.concurrent
        )
        
        # Guardar resultados
        output_path = evaluator.save_results(results)
        
        print(f"\n✅ Evaluación completada exitosamente")
        print(f"💾 Resultados guardados en: {output_path}")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error durante la evaluación: {e}")
        return False


def run_analysis(args, results_file=None):
    """Ejecutar análisis de métricas"""
    
    print("\n📊 Iniciando análisis de métricas...")
    print("⚠️  Análisis de métricas no disponible - MetricsAnalyzer no implementado")
    return True
    
    # TODO: Implementar MetricsAnalyzer
    # try:
    #     # Crear analizador
    #     analyzer = MetricsAnalyzer(results_file)
    #     
    #     if results_file:
    #         analyzer.load_results()
    #     else:
    #         # Buscar el archivo más reciente
    #         results_dir = Path(args.output_dir) / "results"
    #         if not results_dir.exists():
    #             print("❌ No se encontraron resultados para analizar")
    #             return False
    #         
    #         # Buscar archivos con el nuevo formato primero (samples)
    #         result_files = list(results_dir.glob("results_*samples_*.json"))
    #         if not result_files:
    #             # Buscar archivos con el formato anterior (candidatos)
    #             result_files = list(results_dir.glob("results_*cand_*.json"))
    #         if not result_files:
    #             # Buscar archivos con formato muy anterior
    #             result_files = list(results_dir.glob("evaluation_results_*.json"))
    #             
    #         if not result_files:
    #             print("❌ No se encontraron archivos de resultados")
    #             return False
    #             
    #         latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    #         print(f"📂 Usando archivo: {latest_file}")
    #         analyzer.load_results(str(latest_file))
    #     
    #     # Crear DataFrame
    #     df = analyzer.create_dataframe()
    #     print(f"📄 Datos procesados: {len(df)} filas")
    #     
    #     # Generar reporte
    #     report = analyzer.generate_performance_report()
    #     
    #     # Guardar reporte con nombre descriptivo
    #     base_name = analyzer._generate_base_filename()
    #     report_path = Path(args.output_dir) / f"performance_report_{base_name}.json"
    #     report_path.parent.mkdir(parents=True, exist_ok=True)
    #     
    #     with open(report_path, 'w', encoding='utf-8') as f:
    #         json.dump(report, f, indent=2, ensure_ascii=False, default=numpy_json_serializer)
    #     print(f"📋 Reporte guardado en: {report_path}")
    #     
    #     # Exportar CSV detallado con nombre descriptivo
    #     analyzer.export_detailed_csv()
    #     
    #     # Mostrar resumen
    #     print_summary(report, analyzer.compare_agents())
    #     
    #     return True
    #     
    # except Exception as e:
    #     print(f"❌ Error durante el análisis: {e}")
    #     return False


def print_summary(report, comparison_df):
    """Imprimir resumen de resultados con métricas de calidad"""
    
    print("\n" + "="*120)
    print("📊 RESUMEN DE EVALUACIÓN CON GROUND TRUTH")
    print("="*120)
    
    # Estadísticas generales
    general = report["general_statistics"]
    print(f"📈 Estadísticas Generales:")
    print(f"   • Total de ejecuciones: {general['total_executions']}")
    print(f"   • Agentes evaluados: {general['total_agents']}")
    print(f"   • Tiempo total: {general['total_execution_time']:.2f} segundos")
    print(f"   • Costo total estimado: ${general['total_estimated_cost']:.4f}")
    
    if general['total_executions'] > 0:
        print(f"   • Costo promedio por query: ${general['total_estimated_cost']/general['total_executions']:.4f}")
    
    # NUEVAS métricas de calidad
    print(f"\n🎯 MÉTRICAS DE CALIDAD (Ground Truth):")
    print(f"   • Score de calidad promedio: {general['average_quality_score']:.3f}")
    print(f"   • Encontrado en resultados: {general['found_in_results_rate']:.2%}")
    print(f"   • Perfect hits (posición 1): {general['perfect_rate']:.2%}")
    print(f"   • Top 3 hits: {general['top_3_rate']:.2%}")
    print(f"   • Top 5 hits: {general['top_5_rate']:.2%}")
    
    # NUEVAS métricas combinadas
    print(f"\n⚖️  MÉTRICAS COMBINADAS:")
    print(f"   • Éxito técnico únicamente: {general['technical_success_rate']:.2%}")
    print(f"   • Éxito combinado (técnico + calidad): {general['combined_success_rate']:.2%}")
    print(f"   • Nueva tasa de éxito: {general['overall_success_rate']:.2%}")
    
    # Distribución de tiers de calidad
    if 'quality_distribution' in general:
        print(f"\n📊 DISTRIBUCIÓN DE CALIDAD:")
        quality_dist = general['quality_distribution']
        total_samples = sum(quality_dist.values())
        for tier, count in quality_dist.items():
            percentage = count / total_samples * 100 if total_samples > 0 else 0
            tier_names = {
                'perfect': 'Perfecto (pos. 1)',
                'top_3': 'Top 3',
                'top_5': 'Top 5', 
                'found_far': 'Encontrado (pos. >5)',
                'not_found': 'No encontrado',
                'error': 'Error técnico',
                'unknown': 'Desconocido'
            }
            tier_display = tier_names.get(tier, tier)
            print(f"   • {tier_display}: {count} ({percentage:.1f}%)")
    
    # Comparación de agentes actualizada
    print(f"\n🤖 COMPARACIÓN DE AGENTES (Con Métricas de Calidad):")
    print("-" * 120)
    
    for _, row in comparison_df.iterrows():
        agent = row['agent_name']
        print(f"\n{agent.upper()}:")
        
        # Métricas básicas
        print(f"   📊 Básico: {row.get('total_executions', 0):.0f} ejecuciones")
        
        # Métricas técnicas vs combinadas
        tech_success = row.get('technical_success_rate', 0)
        combined_success = row.get('combined_success_rate', row.get('success_rate', 0))
        print(f"   ✅ Éxito: Técnico {tech_success:.2%} | Combinado {combined_success:.2%}")
        
        # Métricas de calidad
        quality_score = row.get('average_quality_score', 0)
        perfect_rate = row.get('perfect_rate', 0)
        top_3_rate = row.get('top_3_rate', 0)
        print(f"   🎯 Calidad: Score {quality_score:.3f} | Perfect {perfect_rate:.2%} | Top-3 {top_3_rate:.2%}")
        
        # Métricas de rendimiento
        avg_time = row.get('avg_execution_time', 0)
        std_time = row.get('std_execution_time', 0)
        print(f"   ⏱️  Tiempo: {avg_time:.2f}s promedio (±{std_time:.2f}s)")
        
        # Métricas de costo
        total_cost = row.get('total_cost', 0)
        avg_cost = row.get('avg_cost', 0)
        print(f"   💰 Costo: ${total_cost:.4f} total (${avg_cost:.4f} promedio)")
        
        # Tokens
        if 'total_tokens' in row:
            total_tokens = row.get('total_tokens', 0)
            avg_tokens = row.get('avg_tokens', 0)
            print(f"   🔤 Tokens: {total_tokens:.0f} total ({avg_tokens:.0f} promedio)")
    
    # Análisis thinking vs regular actualizado
    if 'thinking_vs_regular_analysis' in report:
        thinking_analysis = report["thinking_vs_regular_analysis"]
        print(f"\n🧠 COMPARACIÓN THINKING VS REGULAR (Con Calidad):")
        
        thinking_stats = thinking_analysis['agents_with_thinking']
        regular_stats = thinking_analysis['agents_without_thinking']
        
        # Validar que hay datos para mostrar
        if thinking_stats['count'] > 0 or regular_stats['count'] > 0:
            print(f"Agentes con thinking: {thinking_stats['count']}")
            if thinking_stats['count'] > 0:
                print(f"  • Éxito técnico: {thinking_stats.get('technical_success_rate', 0):.2%}")
                print(f"  • Score de calidad: {thinking_stats.get('average_quality_score', 0):.3f}")
                print(f"  • Éxito combinado: {thinking_stats.get('combined_success_rate', 0):.2%}")
                print(f"  • Costo promedio: ${thinking_stats.get('avg_cost_per_query', 0):.4f}")
                print(f"  • Tiempo promedio: {thinking_stats.get('avg_execution_time', 0):.2f}s")
            else:
                print(f"  • No hay datos disponibles")
            
            print(f"Agentes sin thinking: {regular_stats['count']}")
            if regular_stats['count'] > 0:
                print(f"  • Éxito técnico: {regular_stats.get('technical_success_rate', 0):.2%}")
                print(f"  • Score de calidad: {regular_stats.get('average_quality_score', 0):.3f}")
                print(f"  • Éxito combinado: {regular_stats.get('combined_success_rate', 0):.2%}")
                print(f"  • Costo promedio: ${regular_stats.get('avg_cost_per_query', 0):.4f}")
                print(f"  • Tiempo promedio: {regular_stats.get('avg_execution_time', 0):.2f}s")
            else:
                print(f"  • No hay datos disponibles")
            
            # Calcular mejoras solo si ambos tienen datos
            if thinking_stats['count'] > 0 and regular_stats['count'] > 0:
                quality_improvement = thinking_stats.get('average_quality_score', 0) - regular_stats.get('average_quality_score', 0)
                combined_improvement = thinking_stats.get('combined_success_rate', 0) - regular_stats.get('combined_success_rate', 0)
                cost_increase = thinking_stats.get('avg_cost_per_query', 0) - regular_stats.get('avg_cost_per_query', 0)
                
                print(f"\n💡 ANÁLISIS DE ROI (Thinking Models):")
                print(f"   • Mejora en calidad: {quality_improvement:+.3f}")
                print(f"   • Mejora en éxito combinado: {combined_improvement:+.2%}")
                print(f"   • Incremento en costo: ${cost_increase:+.4f}")
                
                if cost_increase > 0:
                    roi_quality = quality_improvement / cost_increase
                    roi_success = (combined_improvement * 100) / cost_increase
                    print(f"   • ROI calidad: {roi_quality:.2f} puntos por $")
                    print(f"   • ROI éxito: {roi_success:.2f} pp de éxito por $")
        else:
            print(f"  • No hay datos suficientes para comparación")
    
    # Análisis por dificultad si está disponible
    if 'difficulty_level_analysis' in report and report['difficulty_level_analysis']:
        print(f"\n📈 ANÁLISIS POR NIVEL DE DIFICULTAD:")
        difficulty_analysis = report['difficulty_level_analysis']
        
        for difficulty, stats in difficulty_analysis.items():
            print(f"  {difficulty.upper()}:")
            print(f"    • Muestras: {stats['total_samples']}")
            print(f"    • Éxito combinado: {stats['combined_success_rate']:.2%}")
            print(f"    • Score calidad: {stats['average_quality_score']:.3f}")
            print(f"    • Perfect rate: {stats['perfect_rate']:.2%}")
    
    print(f"\n📋 NUEVA DEFINICIÓN DE ÉXITO:")
    print(f"   ✅ Éxito = Ejecución técnica exitosa AND Score de calidad > 0.0")
    print(f"   📊 Sistema de scoring por posición:")
    print(f"      • Posición 1: 1.0 puntos")
    print(f"      • Posiciones 1-3: 0.8 puntos") 
    print(f"      • Posiciones 1-5: 0.6 puntos")
    print(f"      • Más de 5 posiciones: 0.3 puntos")
    print(f"      • No encontrado: 0.0 puntos")
    
    print("\n" + "="*120)


def main():
    """Función principal"""
    
    args = parse_arguments()
    
    print("🎯 Sistema de Evaluación de Agentes PyCiudad")
    print("=" * 50)
    
    success = True
    results_file = None
    
    # Ejecutar evaluación si no es solo análisis
    if not args.analyze_only:
        results_file = asyncio.run(run_evaluation(args))
        if not results_file:
            success = False
    
    # Ejecutar análisis
    if success:
        analysis_success = run_analysis(args, results_file)
        if not analysis_success:
            success = False
    
    # Resultado final
    if success:
        print("\n🎉 Proceso completado exitosamente")
        print(f"📁 Resultados en: {args.output_dir}/")
    else:
        print("\n❌ Proceso terminado con errores")
        sys.exit(1)


if __name__ == "__main__":
    main() 
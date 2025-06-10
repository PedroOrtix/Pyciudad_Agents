#!/usr/bin/env python3
"""
Script de Evaluación Local de Agentes PyCiudad

Ejecuta evaluaciones de agentes localmente sin depender del servicio LangGraph.
Útil para evitar problemas de rate limiting del servidor langgraph dev.
"""

import asyncio
import sys
import argparse
import json
from pathlib import Path

# Importar el sistema de evaluación local
from evaluation.core.agent_evaluation_system_local import AgentEvaluationSystemLocal

def numpy_json_serializer(obj):
    """Serializador JSON personalizado para numpy"""
    import numpy as np
    
    if hasattr(obj, 'dtype'):
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if str(type(obj)).startswith('<class \'numpy.'):
        if 'int' in str(type(obj)):
            return int(obj)
        elif 'float' in str(type(obj)):
            return float(obj)
        elif 'bool' in str(type(obj)):
            return bool(obj)
    if callable(obj):
        return f"<callable: {str(obj)}>"
    if isinstance(obj, set):
        return list(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def parse_arguments():
    """Configurar argumentos de línea de comandos"""
    parser = argparse.ArgumentParser(
        description="Evaluación LOCAL de Agentes PyCiudad",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Evaluación rápida con 25 muestras BALANCEADAS
  python -m evaluation.scripts.run_evaluation_local --samples 25

  # Evaluación con estrategia stratified (equilibrada) - RECOMENDADO
  python -m evaluation.scripts.run_evaluation_local --samples 100 --balance-strategy stratified --max-samples-per-type 20

  # Evaluación sin balanceo (dataset original)
  python -m evaluation.scripts.run_evaluation_local --samples 100 --no-balance

  # Evaluación con undersample (todos los tipos al mismo nivel)
  python -m evaluation.scripts.run_evaluation_local --samples 200 --balance-strategy undersample --min-samples-per-type 15

  # Evaluación proporcional pero limitada
  python -m evaluation.scripts.run_evaluation_local --samples 300 --balance-strategy proportional --max-samples-per-type 30

  # Solo análisis de resultados existentes
  python -m evaluation.scripts.run_evaluation_local --analyze-only

  # Evaluación solo del agente_base con dataset balanceado
  python -m evaluation.scripts.run_evaluation_local --samples 50 --agents agent_base --max-samples-per-type 10

  # Evaluación completa de todos los agentes con dataset balanceado
  python -m evaluation.scripts.run_evaluation_local --full-dataset --concurrent 5 --balance-strategy stratified

  # Evaluación con configuración robusta de red (para conexiones inestables)
  python -m evaluation.scripts.run_evaluation_local --samples 100 --network-wait 60 --max-retries 10

  # Reanudar evaluación desde checkpoint con balanceo
  python -m evaluation.scripts.run_evaluation_local --resume-from my_evaluation_id --balance-strategy stratified

ESTRATEGIAS DE BALANCEO:
  
  - stratified (RECOMENDADO): Toma una cantidad equilibrada de cada tipo de query
    Resultado: Evaluación justa sin sesgo hacia ningún tipo
    
  - undersample: Reduce todos los tipos al mínimo común denominador  
    Resultado: Máxima equidad pero menos samples totales
    
  - proportional: Mantiene proporciones originales pero con límites
    Resultado: Respeta distribución original pero evita extremos
        """
    )
    
    # Argumentos de evaluación
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Número de muestras del dataset a evaluar (default: 100)"
    )
    
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Usar todo el dataset (ignora --samples)"
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/datasets/dataset_generado_final.json",
        help="Ruta al dataset de evaluación (default: data/datasets/dataset_generado_final.json)"
    )
    
    # NUEVOS ARGUMENTOS DE BALANCEO
    parser.add_argument(
        "--balance-dataset",
        action="store_true",
        default=True,
        help="Balancear automáticamente el dataset por tipo de query y dificultad (default: True)"
    )
    
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Desactivar balanceo automático (usar dataset original)"
    )
    
    parser.add_argument(
        "--min-samples-per-type",
        type=int,
        default=10,
        help="Mínimo número de samples por tipo de query (default: 10)"
    )
    
    parser.add_argument(
        "--max-samples-per-type",
        type=int,
        default=50,
        help="Máximo número de samples por tipo de query (default: 50)"
    )
    
    parser.add_argument(
        "--balance-strategy",
        choices=["undersample", "stratified", "proportional"],
        default="stratified",
        help="Estrategia de balanceo: undersample, stratified, proportional (default: stratified)"
    )
    
    parser.add_argument(
        "--concurrent",
        type=int,
        default=3,
        help="Número máximo de ejecuciones concurrentes por agente (default: 3)"
    )
    
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Solo realizar análisis sin ejecutar nueva evaluación"
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
    
    # Argumentos de robustez de red (NUEVOS)
    parser.add_argument(
        "--network-wait",
        type=int,
        default=30,
        help="Minutos a esperar por conectividad cuando se pierde la red (default: 30)"
    )
    
    parser.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Número máximo de reintentos ante fallos de red (default: 5)"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        help="ID de evaluación desde la cual reanudar (usar checkpoints)"
    )
    
    parser.add_argument(
        "--test-connectivity",
        action="store_true",
        help="Solo testear conectividad y salir"
    )
    
    return parser.parse_args()


def analyze_dataset_distribution(dataset_path):
    """Analizar la distribución de tipos de query y dificultad en el dataset"""
    
    print("📊 Analizando distribución del dataset...")
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Contar por tipo y dificultad
    distribution = {}
    total_samples = len(dataset)
    
    for sample in dataset:
        query_type = sample.get('query_type', 'unknown')
        difficulty = sample.get('difficulty_level', 'unknown')
        key = f"{query_type}_{difficulty}"
        
        if key not in distribution:
            distribution[key] = []
        distribution[key].append(sample)
    
    # Mostrar distribución
    print(f"📈 Distribución encontrada ({total_samples} samples totales):")
    sorted_dist = sorted(distribution.items(), key=lambda x: len(x[1]), reverse=True)
    
    for type_diff, samples in sorted_dist:
        count = len(samples)
        percentage = (count / total_samples) * 100
        print(f"   {type_diff}: {count} samples ({percentage:.1f}%)")
    
    return distribution


def balance_dataset(distribution, strategy="stratified", min_samples=10, max_samples=50, target_total=None):
    """
    Balancear el dataset según la estrategia especificada
    
    Estrategias:
    - undersample: Reducir todos los tipos al mínimo común
    - stratified: Tomar una cantidad equilibrada de cada tipo
    - proportional: Mantener proporciones pero limitar extremos
    """
    
    print(f"⚖️  Aplicando estrategia de balanceo: {strategy}")
    
    # Filtrar tipos con suficientes muestras
    valid_types = {k: v for k, v in distribution.items() if len(v) >= min_samples}
    excluded_types = {k: len(v) for k, v in distribution.items() if len(v) < min_samples}
    
    if excluded_types:
        print(f"   ⚠️  Tipos excluidos por pocas muestras: {excluded_types}")
    
    if not valid_types:
        print("   ❌ No hay tipos suficientes para balancear")
        return []
    
    balanced_samples = []
    
    if strategy == "undersample":
        # Usar el mínimo común denominador
        min_count = min(len(samples) for samples in valid_types.values())
        samples_per_type = min(min_count, max_samples)
        
        print(f"   📉 Undersample: {samples_per_type} samples por tipo")
        
        for type_name, samples in valid_types.items():
            selected = samples[:samples_per_type]
            balanced_samples.extend(selected)
            print(f"      {type_name}: {len(selected)} samples")
    
    elif strategy == "stratified":
        # Distribución equilibrada respetando límites
        num_types = len(valid_types)
        if target_total:
            samples_per_type = min(target_total // num_types, max_samples)
        else:
            samples_per_type = max_samples
        
        print(f"   📊 Stratified: {samples_per_type} samples por tipo")
        
        for type_name, samples in valid_types.items():
            available = len(samples)
            to_take = min(samples_per_type, available)
            
            # Muestreo aleatorio para mayor diversidad
            import random
            selected = random.sample(samples, to_take)
            balanced_samples.extend(selected)
            print(f"      {type_name}: {to_take}/{available} samples")
    
    elif strategy == "proportional":
        # Mantener proporciones pero con límites
        total_original = sum(len(samples) for samples in valid_types.values())
        if not target_total:
            target_total = min(total_original, len(valid_types) * max_samples)
        
        print(f"   📈 Proportional: {target_total} samples totales")
        
        for type_name, samples in valid_types.items():
            original_count = len(samples)
            proportion = original_count / total_original
            target_count = int(target_total * proportion)
            
            # Aplicar límites
            final_count = max(min_samples, min(target_count, max_samples, original_count))
            
            import random
            selected = random.sample(samples, final_count)
            balanced_samples.extend(selected)
            print(f"      {type_name}: {final_count} samples ({proportion:.1%} proporción)")
    
    print(f"   ✅ Dataset balanceado: {len(balanced_samples)} samples totales")
    return balanced_samples


def prepare_balanced_dataset(dataset_path, args):
    """Preparar dataset balanceado según configuración"""
    
    # Si no hay balanceo, usar dataset original
    if args.no_balance or not args.balance_dataset:
        print("📋 Usando dataset original (sin balanceo)")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Aplicar límite de samples si se especifica
        if not args.full_dataset and args.samples:
            dataset = dataset[:args.samples]
            print(f"   📏 Limitado a {len(dataset)} samples")
        
        return dataset
    
    # Analizar distribución
    distribution = analyze_dataset_distribution(dataset_path)
    
    # Determinar target total
    target_total = None
    if not args.full_dataset and args.samples:
        target_total = args.samples
    
    # Balancear
    balanced_samples = balance_dataset(
        distribution=distribution,
        strategy=args.balance_strategy,
        min_samples=args.min_samples_per_type,
        max_samples=args.max_samples_per_type,
        target_total=target_total
    )
    
    if not balanced_samples:
        print("❌ Error en el balanceo, usando dataset original")
        with open(dataset_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    # Mezclar para evitar orden por tipo
    import random
    random.shuffle(balanced_samples)
    
    print(f"✅ Dataset balanceado preparado: {len(balanced_samples)} samples")
    return balanced_samples


async def run_evaluation_local_main(args):
    """Ejecutar evaluación local de agentes"""
    
    # Testear conectividad si se solicita
    if args.test_connectivity:
        from evaluation.core.network_resilience import test_connectivity
        test_connectivity()
        return True
    
    print("🚀 Iniciando evaluación LOCAL de agentes...")
    print(f"📊 Configuración:")
    print(f"   - Muestras: {'Todo el dataset' if args.full_dataset else args.samples}")
    print(f"   - Concurrencia: {args.concurrent}")
    print(f"   - Modo: EJECUCIÓN LOCAL (sin LangGraph server)")
    print(f"   - Agentes: {', '.join(args.agents)}")
    
    # Información de balanceo
    if not args.no_balance and args.balance_dataset:
        print(f"⚖️  Configuración de balanceo:")
        print(f"   - Estrategia: {args.balance_strategy}")
        print(f"   - Min samples por tipo: {args.min_samples_per_type}")
        print(f"   - Max samples por tipo: {args.max_samples_per_type}")
    else:
        print(f"📋 Balanceo: DESACTIVADO (dataset original)")
    
    print(f"🛡️  Configuración de robustez de red:")
    print(f"   - Espera por conectividad: {args.network_wait} min")
    print(f"   - Máximo reintentos: {args.max_retries}")
    if args.resume_from:
        print(f"   - Reanudar desde: {args.resume_from}")
    
    # Verificar que el dataset existe
    if not Path(args.dataset).exists():
        print(f"❌ Dataset no encontrado: {args.dataset}")
        return False
    
    # Preparar dataset balanceado
    print(f"\n📂 Procesando dataset: {args.dataset}")
    balanced_dataset = prepare_balanced_dataset(args.dataset, args)
    
    if not balanced_dataset:
        print("❌ Error preparando dataset balanceado")
        return False
    
    # Crear evaluador local con dataset balanceado
    evaluator = AgentEvaluationSystemLocal(dataset_path=args.dataset)
    
    # Override del dataset con la versión balanceada
    evaluator.dataset = balanced_dataset
    
    try:
        # Ejecutar evaluación (el max_samples ya está aplicado en el balanceado)
        max_samples = len(balanced_dataset)
        
        # Configurar parámetros de robustez de red
        evaluation_id = args.resume_from
        
        results = await evaluator.evaluate_all_agents_local(
            max_samples=max_samples,
            max_concurrent=args.concurrent,
            agents_to_evaluate=args.agents,
            evaluation_id=evaluation_id,
            network_wait_minutes=args.network_wait,
            max_retries=args.max_retries
        )
        
        # Guardar resultados
        output_path = evaluator.save_results_local(results)
        
        print(f"\n✅ Evaluación LOCAL completada exitosamente")
        print(f"💾 Resultados guardados en: {output_path}")
        print(f"📊 Logs de red disponibles en: evaluation/network_resilience.log")
        
        return output_path
        
    except Exception as e:
        print(f"❌ Error durante la evaluación LOCAL: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_analysis_local(args, results_file=None):
    """Ejecutar análisis de métricas usando resultados locales"""
    
    print("\n📊 Iniciando análisis de métricas...")
    print("⚠️  Análisis de métricas no disponible - MetricsAnalyzer no implementado")
    return True
    
    # TODO: Implementar MetricsAnalyzer
    # try:
    #     # Crear analizador
    #     from evaluation.analysis.metrics_analyzer import MetricsAnalyzer
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
    #         # Buscar archivos locales primero
    #         result_files = list(results_dir.glob("results_local_*samples_*.json"))
    #         if not result_files:
    #             # Buscar archivos normales si no hay locales
    #             result_files = list(results_dir.glob("results_*samples_*.json"))
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
    #     report_path = Path(args.output_dir) / f"performance_report_local_{base_name}.json"
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
    #     import traceback
    #     traceback.print_exc()
    #     return False


def print_summary(report, comparison_df):
    """Imprimir resumen de resultados"""
    
    print("\n" + "="*60)
    print("📊 RESUMEN DE EVALUACIÓN LOCAL")
    print("="*60)
    
    # Información general
    agents = list(report.keys())
    print(f"🤖 Agentes evaluados: {', '.join(agents)}")
    
    if not agents:
        print("❌ No se encontraron datos de agentes")
        return
    
    # Métricas por agente
    for agent in agents:
        agent_data = report[agent]
        print(f"\n🤖 {agent.upper()}:")
        print(f"   ✅ Éxito combinado: {agent_data.get('combined_success_rate', 0):.1%}")
        print(f"   🎯 Calidad promedio: {agent_data.get('average_quality_score', 0):.2f}")
        print(f"   🥇 Hits perfectos: {agent_data.get('perfect_rate', 0):.1%}")
        print(f"   ⏱️  Tiempo promedio: {agent_data.get('avg_execution_time', 0):.2f}s")
        
        # Mostrar distribución por tier
        tier_dist = agent_data.get('tier_distribution', {})
        if tier_dist:
            print(f"   📈 Distribución:")
            for tier, count in tier_dist.items():
                print(f"      - {tier}: {count}")
    
    print("\n" + "="*60)


def main():
    """Función principal"""
    
    args = parse_arguments()
    
    print("🎯 Sistema de Evaluación LOCAL de Agentes PyCiudad")
    print("=" * 60)
    
    success = True
    results_file = None
    
    # Ejecutar evaluación si no es solo análisis
    if not args.analyze_only:
        results_file = asyncio.run(run_evaluation_local_main(args))
        if not results_file:
            success = False
    
    # Ejecutar análisis
    if success:
        analysis_success = run_analysis_local(args, results_file)
        if not analysis_success:
            success = False
    
    # Resultado final
    if success:
        print("\n🎉 Proceso LOCAL completado exitosamente")
        print(f"📁 Resultados en: {args.output_dir}/")
    else:
        print("\n❌ Proceso LOCAL terminado con errores")
        sys.exit(1)


if __name__ == "__main__":
    main() 
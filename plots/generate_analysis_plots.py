#!/usr/bin/env python3
"""
Script para generar análisis gráfico completo de los resultados de agentes.

Este script lee DIRECTAMENTE un archivo JSON de resultados específico y calcula
todas las estadísticas desde cero, sin depender de performance reports.

Uso: python generate_analysis_plots.py [archivo_resultados.json]

Autor: Análisis automático de rendimiento de agentes
Fecha: Generado automáticamente
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import sys
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# Importaciones adicionales para análisis avanzado
try:
    from scipy import stats
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import pdist, squareform
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Algunas funcionalidades avanzadas no estarán disponibles: {e}")
    print("   Para instalar: pip install scipy scikit-learn")
    ADVANCED_ANALYSIS_AVAILABLE = False

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11

class AgentAnalysisPlotter:
    """
    Clase principal para generar análisis gráfico de resultados de agentes.
    
    Funcionalidades:
    - Carga datos DIRECTAMENTE desde un archivo JSON de resultados específico
    - Calcula TODAS las estadísticas desde cero
    - Genera plots comparativos entre agentes
    - Crea análisis individuales por agente
    """
    
    def __init__(self, results_file: str = None):
        # Detectar si estamos en plots/ o en la raíz del proyecto
        current_dir = Path.cwd()
        if current_dir.name == 'plots':
            self.base_dir = current_dir.parent
        else:
            self.base_dir = current_dir
            
        self.results_dir = self.base_dir / "evaluation" / "results"
        self.plots_dir = self.base_dir / "plots"
        
        # Archivo de resultados específico
        self.results_file = results_file
        
        # Datos cargados
        self.raw_data = {}
        self.calculated_stats = {}
        
        print(f"📁 Directorio base: {self.base_dir}")
        print(f"📁 Directorio de resultados: {self.results_dir}")
        
    def load_data(self):
        """
        Carga datos DIRECTAMENTE desde el archivo JSON de resultados especificado.
        
        Returns:
            bool: True si se cargaron datos exitosamente
        """
        print("\n📊 Cargando datos de resultados...")
        
        # Si no se especifica archivo, buscar el más reciente
        if not self.results_file:
            result_files = list(self.results_dir.glob("results_local_*samples*.json"))
            if not result_files:
                print("   ❌ No se encontraron archivos de resultados")
                return False
            
            # Tomar el más reciente
            self.results_file = max(result_files, key=lambda x: x.stat().st_mtime).name
            print(f"   🔍 Auto-detectado archivo más reciente: {self.results_file}")
        
        # Construir ruta completa
        if not str(self.results_file).startswith('/'):
            results_path = self.results_dir / self.results_file
        else:
            results_path = Path(self.results_file)
        
        # Verificar que existe
        if not results_path.exists():
            print(f"   ❌ No existe el archivo: {results_path}")
            return False
        
        print(f"   📂 Cargando archivo: {results_path.name}")
        print(f"   📏 Tamaño: {results_path.stat().st_size / 1024:.1f} KB")
        
        try:
            with open(results_path, 'r', encoding='utf-8') as f:
                self.raw_data = json.load(f)
            
            print(f"   ✅ Archivo cargado exitosamente")
            
            # Verificar estructura
            if 'results_by_agent' not in self.raw_data:
                print("   ❌ El archivo no tiene la estructura esperada (falta 'results_by_agent')")
                return False
            
            agents = list(self.raw_data['results_by_agent'].keys())
            print(f"   └── Agentes encontrados: {', '.join(agents)}")
            
            # Calcular estadísticas
            self._calculate_statistics()
            
            return True
            
        except Exception as e:
            print(f"   ❌ Error cargando archivo: {e}")
            return False
    
    def _calculate_statistics(self):
        """
        Calcula TODAS las estadísticas desde cero a partir de los datos raw.
        """
        print("   🧮 Calculando estadísticas desde datos raw...")
        
        self.calculated_stats = {
            'agents_analysis': {},
            'thinking_vs_regular_analysis': {},
            'general_statistics': {}
        }
        
        agents_data = {}
        total_executions = 0
        total_success = 0
        total_quality = 0
        thinking_agents = []
        regular_agents = []
        
        # Procesar cada agente
        for agent_name, results in self.raw_data['results_by_agent'].items():
            print(f"   └── Procesando {agent_name}: {len(results)} ejecuciones")
            
            agent_stats = self._calculate_agent_statistics(agent_name, results)
            agents_data[agent_name] = agent_stats
            
            # Acumular para estadísticas generales
            total_executions += agent_stats['total_executions']
            total_success += agent_stats['combined_successful_executions']
            total_quality += agent_stats['total_quality_sum']
            
            # Clasificar thinking vs regular
            if agent_stats.get('has_thinking_model', False):
                thinking_agents.append(agent_name)
            else:
                regular_agents.append(agent_name)
        
        # Guardar estadísticas por agente
        self.calculated_stats['agents_analysis'] = agents_data
        
        # Calcular estadísticas generales
        self.calculated_stats['general_statistics'] = {
            'total_executions': total_executions,
            'total_agents': len(agents_data),
            'combined_success_rate': total_success / total_executions if total_executions > 0 else 0,
            'average_quality_score': total_quality / total_executions if total_executions > 0 else 0
        }
        
        # Calcular thinking vs regular
        self._calculate_thinking_vs_regular(agents_data, thinking_agents, regular_agents)
        
        print(f"   ✅ Estadísticas calculadas para {len(agents_data)} agentes")
    
    def _calculate_agent_statistics(self, agent_name: str, results: List[Dict]) -> Dict:
        """
        Calcula estadísticas completas para un agente específico.
        """
        stats = {
            'agent_name': agent_name,
            'total_executions': len(results),
            'technical_successful_executions': 0,
            'combined_successful_executions': 0,
            'perfect_hits': 0,
            'top_3_hits': 0,
            'top_5_hits': 0,
            'found_in_results_count': 0,
            'execution_times': [],
            'quality_scores': [],
            'tier_distribution': {},
            'total_quality_sum': 0,
            'has_thinking_model': False
        }
        
        # Detectar si usa thinking model (basado en metadata o tiempo promedio)
        avg_time = np.mean([r.get('execution_time_seconds', 0) for r in results if r.get('execution_time_seconds')])
        stats['has_thinking_model'] = avg_time > 10  # Heurística: thinking models son más lentos
        
        # Procesar cada resultado
        for result in results:
            # Technical success
            if result.get('technical_success', False):
                stats['technical_successful_executions'] += 1
            
            # Combined success
            if result.get('combined_success', False) or result.get('success', False):
                stats['combined_successful_executions'] += 1
            
            # Quality score
            quality = result.get('quality_score', 0)
            stats['quality_scores'].append(quality)
            stats['total_quality_sum'] += quality
            
            # Execution time
            exec_time = result.get('execution_time_seconds', 0)
            if exec_time > 0:
                stats['execution_times'].append(exec_time)
            
            # Found in results
            if result.get('found_in_results', False):
                stats['found_in_results_count'] += 1
            
            # Scoring tiers
            tier = result.get('scoring_tier', 'unknown')
            stats['tier_distribution'][tier] = stats['tier_distribution'].get(tier, 0) + 1
            
            # Perfect/Top hits
            if tier == 'perfect':
                stats['perfect_hits'] += 1
                stats['top_3_hits'] += 1
                stats['top_5_hits'] += 1
            elif tier == 'top_3':
                stats['top_3_hits'] += 1
                stats['top_5_hits'] += 1
            elif tier == 'top_5':
                stats['top_5_hits'] += 1
        
        # Calcular rates y promedios
        total = stats['total_executions']
        if total > 0:
            stats['technical_success_rate'] = stats['technical_successful_executions'] / total
            stats['combined_success_rate'] = stats['combined_successful_executions'] / total
            stats['perfect_rate'] = stats['perfect_hits'] / total
            stats['top_3_rate'] = stats['top_3_hits'] / total
            stats['top_5_rate'] = stats['top_5_hits'] / total
            stats['found_in_results_rate'] = stats['found_in_results_count'] / total
            stats['average_quality_score'] = stats['total_quality_sum'] / total
        
        # Estadísticas de tiempo
        if stats['execution_times']:
            stats['avg_execution_time'] = np.mean(stats['execution_times'])
            stats['std_execution_time'] = np.std(stats['execution_times'])
            stats['min_execution_time'] = np.min(stats['execution_times'])
            stats['max_execution_time'] = np.max(stats['execution_times'])
        
        return stats
    
    def _calculate_thinking_vs_regular(self, agents_data: Dict, thinking_agents: List, regular_agents: List):
        """
        Calcula estadísticas comparativas entre agentes thinking y regulares.
        """
        thinking_stats = {
            'count': len(thinking_agents),
            'total_executions': 0,
            'technical_success_rate': 0,
            'combined_success_rate': 0,
            'average_quality_score': 0,
            'perfect_rate': 0,
            'avg_execution_time': 0
        }
        
        regular_stats = {
            'count': len(regular_agents),
            'total_executions': 0,
            'technical_success_rate': 0,
            'combined_success_rate': 0,
            'average_quality_score': 0,
            'perfect_rate': 0,
            'avg_execution_time': 0
        }
        
        # Agregar estadísticas de thinking agents
        if thinking_agents:
            for agent in thinking_agents:
                data = agents_data[agent]
                thinking_stats['total_executions'] += data['total_executions']
                thinking_stats['technical_success_rate'] += data.get('technical_success_rate', 0)
                thinking_stats['combined_success_rate'] += data.get('combined_success_rate', 0)
                thinking_stats['average_quality_score'] += data.get('average_quality_score', 0)
                thinking_stats['perfect_rate'] += data.get('perfect_rate', 0)
                thinking_stats['avg_execution_time'] += data.get('avg_execution_time', 0)
            
            # Promediar
            count = len(thinking_agents)
            for key in ['technical_success_rate', 'combined_success_rate', 'average_quality_score', 'perfect_rate', 'avg_execution_time']:
                thinking_stats[key] /= count
        
        # Agregar estadísticas de regular agents
        if regular_agents:
            for agent in regular_agents:
                data = agents_data[agent]
                regular_stats['total_executions'] += data['total_executions']
                regular_stats['technical_success_rate'] += data.get('technical_success_rate', 0)
                regular_stats['combined_success_rate'] += data.get('combined_success_rate', 0)
                regular_stats['average_quality_score'] += data.get('average_quality_score', 0)
                regular_stats['perfect_rate'] += data.get('perfect_rate', 0)
                regular_stats['avg_execution_time'] += data.get('avg_execution_time', 0)
            
            # Promediar
            count = len(regular_agents)
            for key in ['technical_success_rate', 'combined_success_rate', 'average_quality_score', 'perfect_rate', 'avg_execution_time']:
                regular_stats[key] /= count
        
        self.calculated_stats['thinking_vs_regular_analysis'] = {
            'agents_with_thinking': thinking_stats,
            'agents_without_thinking': regular_stats
        }
    
    def generate_comparison_plots(self):
        """
        Genera plots de comparación entre agentes.
        
        Plots generados:
        1. Barras agrupadas de métricas principales
        2. Distribución de tiers de calidad
        3. Heatmap de rendimiento relativo
        4. Radar chart multimétrico
        5. Pareto frontier analysis
        """
        print("\n🔄 Generando plots de comparación entre agentes...")
        comparison_dir = self.plots_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        if not self.calculated_stats.get('agents_analysis'):
            print("   ⚠️  No hay datos de agentes para comparar")
            return
        
        agents_data = self.calculated_stats['agents_analysis']
        
        # 1. PLOT: Métricas principales por agente (Barras agrupadas)
        self._plot_main_metrics_comparison(agents_data, comparison_dir)
        
        # 2. PLOT: Distribución de calidad (si hay datos detallados)
        if self.raw_data.get('results_by_agent'):
            self._plot_quality_distribution(comparison_dir)
        
        # 3. PLOT: Heatmap de rendimiento relativo
        self._plot_performance_heatmap(agents_data, comparison_dir)
        
        # 4. PLOT: Radar chart multimétrico
        self._plot_radar_chart(agents_data, comparison_dir)
        
        # 5. PLOT: Pareto frontier analysis
        self._plot_pareto_frontier(agents_data, comparison_dir)
        
        print("   ✅ Plots de comparación generados en /plots/comparisons/")
    
    def _plot_main_metrics_comparison(self, agents_data: Dict, output_dir: Path):
        """
        PLOT 1: Barras agrupadas de métricas principales
        
        Propósito: Comparación visual rápida de métricas clave entre agentes
        Métricas: combined_success_rate, perfect_rate, top_3_rate, average_quality_score
        """
        metrics = ['combined_success_rate', 'perfect_rate', 'top_3_rate', 'average_quality_score']
        agents = list(agents_data.keys())
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(agents))
        width = 0.2
        
        colors = ['#3498db', '#e74c3c', '#f39c12', '#27ae60']
        
        for i, metric in enumerate(metrics):
            values = [agents_data[agent].get(metric, 0) for agent in agents]
            bars = ax.bar(x + i * width, values, width, label=metric.replace('_', ' ').title(), 
                         color=colors[i], alpha=0.8)
            
            # Agregar valores en las barras
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Agentes', fontweight='bold')
        ax.set_ylabel('Puntuación (0-1)', fontweight='bold')
        ax.set_title('📊 Comparación de Métricas Principales por Agente\n' + 
                    'Higher is better para todas las métricas', fontweight='bold', pad=20)
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels([agent.replace('agent_', '').title() for agent in agents])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "01_main_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Crear archivo de documentación
        doc_text = """
        📊 PLOT: Comparación de Métricas Principales por Agente
        
        QUÉ MUESTRA:
        - Combined Success Rate: % de queries resueltas exitosamente
        - Perfect Rate: % de queries con respuesta perfecta (posición 1)
        - Top 3 Rate: % de queries con respuesta en top 3
        - Average Quality Score: Puntuación promedio de calidad (0-1)
        
        INTERPRETACIÓN:
        - Barras más altas = mejor rendimiento
        - Comparar alturas entre agentes para ver quién es mejor en cada métrica
        - Gaps entre perfect_rate y top_3_rate indican dependencia de posición exacta
        
        USO EN PRESENTACIÓN:
        - Slide principal para mostrar rendimiento general
        - Ideal para destacar qué agente es mejor overall
        """
        
        with open(output_dir / "01_main_metrics_comparison_DOC.txt", "w") as f:
            f.write(doc_text)
    
    def _plot_quality_distribution(self, output_dir: Path):
        """
        PLOT 4: Distribución de niveles de calidad por agente
        
        Propósito: Mostrar cómo se distribuyen los resultados en diferentes tiers
        """
        if not self.raw_data.get('results_by_agent'):
            return
        
        # Extraer distribución de tiers por agente
        agent_distributions = {}
        
        for agent, results in self.raw_data['results_by_agent'].items():
            tiers = {'perfect': 0, 'top_3': 0, 'top_5': 0, 'found_far': 0, 'not_found': 0}
            
            for result in results:
                tier = result.get('scoring_tier', 'not_found')
                if tier in tiers:
                    tiers[tier] += 1
            
            # Convertir a porcentajes
            total = sum(tiers.values())
            if total > 0:
                agent_distributions[agent] = {k: v/total*100 for k, v in tiers.items()}
        
        if not agent_distributions:
            return
        
        # Crear stacked bar chart
        fig, ax = plt.subplots(figsize=(14, 8))
        
        agents = list(agent_distributions.keys())
        tiers = ['perfect', 'top_3', 'top_5', 'found_far', 'not_found']
        colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#95a5a6']
        
        bottom = np.zeros(len(agents))
        
        for i, tier in enumerate(tiers):
            values = [agent_distributions[agent].get(tier, 0) for agent in agents]
            bars = ax.bar([agent.replace('agent_', '').title() for agent in agents], 
                         values, bottom=bottom, label=tier.replace('_', ' ').title(),
                         color=colors[i], alpha=0.8)
            
            # Añadir porcentajes en las barras (solo si > 5%)
            for j, (bar, value) in enumerate(zip(bars, values)):
                if value > 5:
                    ax.text(bar.get_x() + bar.get_width()/2., 
                           bottom[j] + value/2,
                           f'{value:.1f}%', ha='center', va='center',
                           fontweight='bold', color='white')
            
            bottom += values
        
        ax.set_ylabel('Porcentaje de Resultados (%)', fontweight='bold')
        ax.set_xlabel('Agentes', fontweight='bold')
        ax.set_title('📈 Distribución de Calidad de Resultados por Agente\n' +
                    'Verde = Mejor, Gris = Peor', fontweight='bold', pad=20)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "04_quality_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = """
        📈 PLOT: Distribución de Calidad de Resultados
        
        QUÉ MUESTRA:
        Porcentaje de queries que cada agente resuelve en cada nivel de calidad
        
        NIVELES DE CALIDAD (de mejor a peor):
        - Perfect (Verde): Respuesta correcta en posición 1
        - Top 3 (Naranja): Respuesta correcta en posiciones 2-3
        - Top 5 (Naranja claro): Respuesta correcta en posiciones 4-5
        - Found Far (Rojo): Respuesta correcta pero en posición >5
        - Not Found (Gris): No encontró la respuesta correcta
        
        INTERPRETACIÓN:
        - Más verde = mejor agente
        - Más gris = peor agente
        - Comparar proporciones entre agentes
        
        USO EN PRESENTACIÓN:
        - Mostrar consistencia de cada agente
        - Identificar si un agente es "todo o nada" vs consistente
        """
        
        with open(output_dir / "04_quality_distribution_DOC.txt", "w") as f:
            f.write(doc_text)
    
    def _plot_performance_heatmap(self, agents_data: Dict, output_dir: Path):
        """
        PLOT 5: Heatmap de rendimiento relativo
        
        Propósito: Vista matricial de todas las métricas normalizadas
        """
        # Preparar datos para heatmap
        metrics = ['combined_success_rate', 'perfect_rate', 'top_3_rate', 'top_5_rate', 
                  'average_quality_score', 'technical_success_rate']
        
        agents = list(agents_data.keys())
        data_matrix = []
        
        for agent in agents:
            row = []
            for metric in metrics:
                value = agents_data[agent].get(metric, 0)
                row.append(value)
            data_matrix.append(row)
        
        df = pd.DataFrame(data_matrix, 
                         index=[agent.replace('agent_', '').title() for agent in agents],
                         columns=[metric.replace('_', ' ').title() for metric in metrics])
        
        # Crear heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(df, annot=True, cmap='RdYlGn', center=0.5, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.3f', ax=ax)
        
        ax.set_title('🔥 Heatmap de Rendimiento por Agente y Métrica\n' +
                    'Verde = Mejor, Rojo = Peor', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / "05_performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = """
        🔥 PLOT: Heatmap de Rendimiento
        
        QUÉ MUESTRA:
        Matriz de calor con todas las métricas normalizadas por agente
        
        COLORES:
        - Verde intenso: Rendimiento excelente (cercano a 1.0)
        - Amarillo: Rendimiento promedio (cercano a 0.5)
        - Rojo: Rendimiento bajo (cercano a 0.0)
        
        INTERPRETACIÓN:
        - Filas = Agentes (comparar horizontalmente)
        - Columnas = Métricas (comparar verticalmente)
        - Buscar patrones: ¿un agente es bueno en todo o especializado?
        
        USO EN PRESENTACIÓN:
        - Vista general rápida de fortalezas/debilidades
        - Identificar métricas donde todos fallan o destacan
        - Comparación visual inmediata entre agentes
        """
        
        with open(output_dir / "05_performance_heatmap_DOC.txt", "w") as f:
            f.write(doc_text)
    
    def generate_individual_plots(self):
        """
        Genera análisis individuales para cada agente.
        
        Para cada agente crea:
        1. Distribución de tiempos de ejecución
        2. Distribución de quality scores
        3. Timeline de rendimiento (si hay datos temporales)
        """
        print("\n📈 Generando plots individuales por agente...")
        
        if not self.raw_data.get('results_by_agent'):
            print("   ⚠️  No hay datos detallados para análisis individual")
            return
        
        for agent, results in self.raw_data['results_by_agent'].items():
            print(f"   └── Procesando {agent}...")
            agent_dir = self.plots_dir / agent
            agent_dir.mkdir(exist_ok=True)
            
            self._plot_individual_performance(agent, results, agent_dir)
    
    def _plot_individual_performance(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        Genera plots individuales para un agente específico.
        """
        # Extraer datos
        execution_times = [r.get('execution_time_seconds', 0) for r in results if r.get('execution_time_seconds')]
        quality_scores = [r.get('quality_score', 0) for r in results if 'quality_score' in r]
        scoring_tiers = [r.get('scoring_tier', 'unknown') for r in results]
        
        if not execution_times:
            return
        
        # Plot 1: Distribución de tiempos
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Histograma de tiempos
        ax1.hist(execution_times, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('Distribución de Tiempos de Ejecución', fontweight='bold')
        ax1.set_xlabel('Tiempo (segundos)')
        ax1.set_ylabel('Frecuencia')
        ax1.axvline(np.mean(execution_times), color='red', linestyle='--', 
                   label=f'Media: {np.mean(execution_times):.1f}s')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot de tiempos
        ax2.boxplot(execution_times, patch_artist=True, 
                   boxprops=dict(facecolor='lightblue', alpha=0.7))
        ax2.set_title('Box Plot - Tiempos de Ejecución', fontweight='bold')
        ax2.set_ylabel('Tiempo (segundos)')
        ax2.grid(True, alpha=0.3)
        
        # Distribución de quality scores
        if quality_scores:
            ax3.hist(quality_scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
            ax3.set_title('Distribución de Quality Scores', fontweight='bold')
            ax3.set_xlabel('Quality Score (0-1)')
            ax3.set_ylabel('Frecuencia')
            ax3.axvline(np.mean(quality_scores), color='red', linestyle='--',
                       label=f'Media: {np.mean(quality_scores):.3f}')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Distribución de tiers
        tier_counts = pd.Series(scoring_tiers).value_counts()
        colors = {'perfect': '#27ae60', 'top_3': '#f39c12', 'top_5': '#e67e22', 
                 'found_far': '#e74c3c', 'not_found': '#95a5a6'}
        
        wedges, texts, autotexts = ax4.pie(tier_counts.values, labels=tier_counts.index,
                                          autopct='%1.1f%%', startangle=90,
                                          colors=[colors.get(tier, '#cccccc') for tier in tier_counts.index])
        ax4.set_title('Distribución de Resultados por Tier', fontweight='bold')
        
        plt.suptitle(f'📊 Análisis Individual: {agent_name.replace("agent_", "").title()}\n' +
                    f'Total de ejecuciones: {len(results)}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_individual_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Estadísticas resumidas
        stats_text = f"""
        📊 ANÁLISIS INDIVIDUAL: {agent_name.upper()}
        
        ESTADÍSTICAS DE TIEMPO:
        - Media: {np.mean(execution_times):.2f} segundos
        - Mediana: {np.median(execution_times):.2f} segundos
        - Desviación estándar: {np.std(execution_times):.2f} segundos
        - Mínimo: {np.min(execution_times):.2f} segundos
        - Máximo: {np.max(execution_times):.2f} segundos
        
        ESTADÍSTICAS DE CALIDAD:
        - Quality score promedio: {np.mean(quality_scores):.3f}
        - Quality score mediano: {np.median(quality_scores):.3f}
        - Desviación estándar: {np.std(quality_scores):.3f}
        
        DISTRIBUCIÓN DE RESULTADOS:
        {tier_counts.to_string()}
        
        INTERPRETACIÓN:
        - Consistencia temporal: {"Alta" if np.std(execution_times) < np.mean(execution_times) * 0.5 else "Baja"}
        - Consistencia de calidad: {"Alta" if np.std(quality_scores) < 0.3 else "Baja"}
        - Tipo de agente: {"Especialista" if tier_counts.get('perfect', 0) / len(results) > 0.7 else "Generalista"}
        """
        
        with open(output_dir / f"{agent_name}_stats.txt", "w") as f:
            f.write(stats_text)
    
    def create_summary_report(self):
        """
        Crea un reporte resumen con insights principales.
        """
        print("\n📋 Generando reporte resumen...")
        
        if not self.calculated_stats.get('agents_analysis'):
            return
        
        agents_data = self.calculated_stats['agents_analysis']
        
        # Encontrar el mejor agente en cada métrica
        best_agents = {}
        metrics = ['combined_success_rate', 'perfect_rate', 'average_quality_score', 'avg_execution_time']
        
        for metric in metrics:
            if metric == 'avg_execution_time':  # Menor es mejor
                best_agent = min(agents_data.items(), key=lambda x: x[1].get(metric, float('inf')))
            else:  # Mayor es mejor
                best_agent = max(agents_data.items(), key=lambda x: x[1].get(metric, 0))
            best_agents[metric] = best_agent
        
        # Calcular insights
        thinking_data = self.calculated_stats.get('thinking_vs_regular_analysis', {})
        
        report = f"""
        🎯 REPORTE EJECUTIVO - ANÁLISIS DE AGENTES
        ==========================================
        
        📊 RESUMEN GENERAL:
        - Total de agentes evaluados: {len(agents_data)}
        - Mejor agente overall: {max(agents_data.items(), key=lambda x: x[1].get('combined_success_rate', 0))[0]}
        - Success rate general: {self.calculated_stats.get('general_statistics', {}).get('combined_success_rate', 0):.1%}
        
        🏆 MEJORES AGENTES POR MÉTRICA:
        - Mayor Success Rate: {best_agents['combined_success_rate'][0]} ({best_agents['combined_success_rate'][1].get('combined_success_rate', 0):.1%})
        - Mayor Perfect Rate: {best_agents['perfect_rate'][0]} ({best_agents['perfect_rate'][1].get('perfect_rate', 0):.1%})
        - Mayor Quality Score: {best_agents['average_quality_score'][0]} ({best_agents['average_quality_score'][1].get('average_quality_score', 0):.3f})
        - Más Rápido: {best_agents['avg_execution_time'][0]} ({best_agents['avg_execution_time'][1].get('avg_execution_time', 0):.1f}s)
        
        🧠 ANÁLISIS THINKING vs REGULAR:
        """
        
        if thinking_data:
            with_thinking = thinking_data.get('agents_with_thinking', {})
            without_thinking = thinking_data.get('agents_without_thinking', {})
            
            speed_ratio = with_thinking.get('avg_execution_time', 1) / without_thinking.get('avg_execution_time', 1)
            success_diff = (with_thinking.get('combined_success_rate', 0) - without_thinking.get('combined_success_rate', 0)) * 100
            
            report += f"""
        - Agentes con thinking: {with_thinking.get('count', 0)}
        - Agentes sin thinking: {without_thinking.get('count', 0)}
        - Thinking es {speed_ratio:.1f}x más lento
        - Thinking tiene {success_diff:+.1f}% de diferencia en success rate
        - Conclusión: {"Thinking vale la pena" if success_diff > 10 else "Regular es más eficiente"}
        """
        
        report += f"""
        
        💡 INSIGHTS CLAVE PARA PRESENTACIÓN:
        1. El agente más equilibrado es: {max(agents_data.items(), key=lambda x: (x[1].get('combined_success_rate', 0) + x[1].get('average_quality_score', 0))/2)[0]}
        2. Trade-off principal: Velocidad vs Precisión
        3. Los modelos thinking no siempre son mejores
        4. Baseline performance es sorprendentemente fuerte
        
        📈 PLOTS GENERADOS:
        - /plots/comparisons/: 5 gráficos de comparación
        - /plots/agent_X/: Análisis individual por agente
        - Cada plot incluye documentación detallada (_DOC.txt)
        
        🎯 RECOMENDACIONES:
        - Para presentaciones: Usar plots 01, 02 y 03
        - Para análisis técnico: Usar plots 04 y 05
        - Para narrativa: Enfocarse en thinking vs regular
        """
        
        with open(self.plots_dir / "EXECUTIVE_SUMMARY.txt", "w") as f:
            f.write(report)
        
        print("   ✅ Reporte ejecutivo creado en /plots/EXECUTIVE_SUMMARY.txt")

    # =================== NUEVOS PLOTS ESPECÍFICOS ===================
    
    def generate_advanced_specific_plots(self):
        """
        Genera plots específicos avanzados para cada agente.
        
        Plots incluyen:
        1. Análisis de tipos de query
        2. Patrones de fallo específicos
        """
        print("\n🎯 Generando plots específicos avanzados por agente...")
        
        if not self.raw_data.get('results_by_agent'):
            print("   ⚠️  No hay datos detallados para análisis específico avanzado")
            return
        
        for agent, results in self.raw_data['results_by_agent'].items():
            print(f"   └── Procesando plots avanzados para {agent}...")
            agent_dir = self.plots_dir / agent
            agent_dir.mkdir(exist_ok=True)
            
            # Plot 1: Análisis por tipos de query
            self._plot_query_type_analysis(agent, results, agent_dir)
            
            # Plot 2: Análisis de patrones de fallo
            self._plot_failure_patterns(agent, results, agent_dir)
    
    def _plot_query_type_analysis(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        PLOT ESPECÍFICO 3: Análisis por tipos de query
        """
        # Extraer información de tipos de query
        query_analysis = {}
        
        for r in results:
            dataset_info = r.get('dataset_info', {})
            query_type = dataset_info.get('query_type', 'unknown')
            difficulty = dataset_info.get('difficulty_level', 'unknown')
            has_errors = dataset_info.get('has_errors', False)
            
            key = f"{query_type}_{difficulty}"
            if key not in query_analysis:
                query_analysis[key] = {
                    'count': 0,
                    'success_count': 0,
                    'quality_sum': 0,
                    'time_sum': 0,
                    'error_count': 0
                }
            
            query_analysis[key]['count'] += 1
            if r.get('combined_success', False):
                query_analysis[key]['success_count'] += 1
            query_analysis[key]['quality_sum'] += r.get('quality_score', 0)
            query_analysis[key]['time_sum'] += r.get('execution_time_seconds', 0)
            if has_errors:
                query_analysis[key]['error_count'] += 1
        
        if not query_analysis or len(query_analysis) <= 1:
            return
        
        # Preparar datos para plotting
        categories = list(query_analysis.keys())
        success_rates = [query_analysis[cat]['success_count'] / query_analysis[cat]['count'] for cat in categories]
        avg_quality = [query_analysis[cat]['quality_sum'] / query_analysis[cat]['count'] for cat in categories]
        avg_time = [query_analysis[cat]['time_sum'] / query_analysis[cat]['count'] for cat in categories]
        counts = [query_analysis[cat]['count'] for cat in categories]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Success rate por tipo
        bars1 = ax1.bar(categories, success_rates, alpha=0.7, color='skyblue')
        ax1.set_title('📊 Success Rate por Tipo de Query', fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.tick_params(axis='x', rotation=45)
        for bar, rate in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{rate:.2f}', ha='center', va='bottom', fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality score por tipo
        bars2 = ax2.bar(categories, avg_quality, alpha=0.7, color='lightgreen')
        ax2.set_title('📊 Quality Score Promedio por Tipo', fontweight='bold')
        ax2.set_ylabel('Quality Score Promedio')
        ax2.tick_params(axis='x', rotation=45)
        for bar, quality in zip(bars2, avg_quality):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{quality:.2f}', ha='center', va='bottom', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Tiempo promedio por tipo
        bars3 = ax3.bar(categories, avg_time, alpha=0.7, color='orange')
        ax3.set_title('📊 Tiempo Promedio por Tipo', fontweight='bold')
        ax3.set_ylabel('Tiempo Promedio (segundos)')
        ax3.tick_params(axis='x', rotation=45)
        for bar, time in zip(bars3, avg_time):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cantidad de samples por tipo
        bars4 = ax4.bar(categories, counts, alpha=0.7, color='purple')
        ax4.set_title('📊 Número de Samples por Tipo', fontweight='bold')
        ax4.set_ylabel('Cantidad de Samples')
        ax4.tick_params(axis='x', rotation=45)
        for bar, count in zip(bars4, counts):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom', fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'🎯 Análisis por Tipos de Query: {agent_name.replace("agent_", "").title()}\n' +
                    'Rendimiento segmentado por características de la consulta', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_query_type_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_failure_patterns(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        PLOT ESPECÍFICO 4: Análisis detallado de patrones de fallo
        """
        # Separar éxitos y fallos
        successes = [r for r in results if r.get('combined_success', False)]
        failures = [r for r in results if not r.get('combined_success', False)]
        
        if not failures:
            print(f"   ⚠️  {agent_name} no tiene fallos para analizar")
            return
        
        # Reorganizado: Solo 2 plots horizontales
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Comparación Quality Score y Tiempo entre éxitos y fallos
        success_qualities = [r.get('quality_score', 0) for r in successes]
        failure_qualities = [r.get('quality_score', 0) for r in failures]
        success_times = [r.get('execution_time_seconds', 0) for r in successes if r.get('execution_time_seconds')]
        failure_times = [r.get('execution_time_seconds', 0) for r in failures if r.get('execution_time_seconds')]
        
        # Box plots combinados
        if success_qualities:
            bp1 = ax1.boxplot([success_qualities, failure_qualities], 
                             labels=['Éxitos', 'Fallos'], 
                             positions=[1, 2], 
                             patch_artist=True,
                             widths=0.6)
            bp1['boxes'][0].set_facecolor('#27ae60')
            bp1['boxes'][0].set_alpha(0.7)
            bp1['boxes'][1].set_facecolor('#e74c3c')
            bp1['boxes'][1].set_alpha(0.7)
        
        ax1.set_title('📊 Quality Score: Éxitos vs Fallos', fontweight='bold')
        ax1.set_ylabel('Quality Score')
        ax1.grid(True, alpha=0.3)
        
        # Añadir estadísticas en el plot
        if success_qualities and failure_qualities:
            success_mean = np.mean(success_qualities)
            failure_mean = np.mean(failure_qualities)
            ax1.text(0.5, 0.95, 
                    f'Media Éxitos: {success_mean:.3f}\nMedia Fallos: {failure_mean:.3f}\nDiferencia: {success_mean - failure_mean:.3f}',
                    transform=ax1.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        # Plot 2: Análisis de fallos por dificultad y candidates
        failure_by_difficulty = {}
        failure_candidates = []
        
        for f in failures:
            difficulty = f.get('dataset_info', {}).get('difficulty_level', 'unknown')
            failure_by_difficulty[difficulty] = failure_by_difficulty.get(difficulty, 0) + 1
            
            candidates = f.get('total_candidates', 0)
            failure_candidates.append(candidates)
        
        # Crear subplot dividido
        if failure_by_difficulty and len(failure_by_difficulty) > 1:
            # Pie chart de dificultad (lado izquierdo)
            difficulties = list(failure_by_difficulty.keys())
            counts = list(failure_by_difficulty.values())
            colors_pie = plt.cm.Reds(np.linspace(0.4, 0.8, len(difficulties)))
            
            # Usar la mitad izquierda para el pie chart
            ax2_left = plt.subplot(1, 4, 3)
            wedges, texts, autotexts = ax2_left.pie(counts, labels=difficulties, autopct='%1.1f%%',
                                                   colors=colors_pie, startangle=90)
            ax2_left.set_title('Fallos por\nDificultad', fontweight='bold', fontsize=10)
            
            # Usar la mitad derecha para el histograma
            ax2_right = plt.subplot(1, 4, 4)
            if failure_candidates:
                ax2_right.hist(failure_candidates, bins=max(5, len(set(failure_candidates))), 
                              alpha=0.7, color='red', edgecolor='black')
                ax2_right.set_title('Candidates en\nFallos', fontweight='bold', fontsize=10)
                ax2_right.set_xlabel('Número de Candidates')
                ax2_right.set_ylabel('Frecuencia')
                ax2_right.grid(True, alpha=0.3)
                
                # Añadir estadística
                mean_candidates = np.mean(failure_candidates)
                ax2_right.axvline(mean_candidates, color='darkred', linestyle='--', 
                                 label=f'Media: {mean_candidates:.1f}')
                ax2_right.legend()
        else:
            # Si no hay suficientes datos de dificultad, solo mostrar candidates
            if failure_candidates:
                ax2.hist(failure_candidates, bins=max(5, len(set(failure_candidates))), 
                        alpha=0.7, color='red', edgecolor='black')
                ax2.set_title('📊 Candidates Encontrados en Fallos', fontweight='bold')
                ax2.set_xlabel('Número de Candidates')
                ax2.set_ylabel('Frecuencia')
                ax2.grid(True, alpha=0.3)
                
                mean_candidates = np.mean(failure_candidates) if failure_candidates else 0
                ax2.axvline(mean_candidates, color='darkred', linestyle='--', 
                           label=f'Media: {mean_candidates:.1f}')
                ax2.legend()
        
        plt.suptitle(f'❌ Análisis de Patrones de Fallo: {agent_name.replace("agent_", "").title()}\n' +
                    f'Total fallos: {len(failures)} de {len(results)} queries ({len(failures)/len(results)*100:.1f}%)', 
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_failure_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    # =================== NUEVOS PLOTS COMPARATIVOS ===================
    
    def generate_advanced_comparison_plots(self):
        """
        Genera plots comparativos avanzados entre agentes.
        
        Plots incluyen:
        1. Radar chart multimétrico
        2. Pareto frontier analysis
        """
        print("\n🏆 Generando plots comparativos avanzados...")
        comparison_dir = self.plots_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        
        if not self.calculated_stats.get('agents_analysis'):
            print("   ⚠️  No hay datos de agentes para comparación avanzada")
            return
        
        agents_data = self.calculated_stats['agents_analysis']
        
        # Plot 1: Radar chart multimétrico
        self._plot_radar_chart(agents_data, comparison_dir)
        
        # Plot 2: Pareto frontier analysis
        self._plot_pareto_frontier(agents_data, comparison_dir)
        
        print("   ✅ Plots comparativos avanzados generados")
    
    def _plot_radar_chart(self, agents_data: Dict, output_dir: Path):
        """
        PLOT COMPARATIVO 1: Radar chart multimétrico
        """
        # Métricas para el radar
        metrics = ['combined_success_rate', 'perfect_rate', 'average_quality_score', 
                  'technical_success_rate', 'found_in_results_rate']
        metric_labels = ['Success Rate', 'Perfect Rate', 'Quality Score', 
                        'Technical Success', 'Found Rate']
        
        agents = list(agents_data.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(agents)))
        
        # Configurar ángulos para el radar
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Cerrar el círculo
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        for i, agent in enumerate(agents):
            values = []
            for metric in metrics:
                value = agents_data[agent].get(metric, 0)
                values.append(value)
            
            values += values[:1]  # Cerrar el círculo
            
            ax.plot(angles, values, 'o-', linewidth=2, label=agent.replace('agent_', '').title(),
                   color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        # Configurar el radar
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.title('🎯 Radar Chart: Comparación Multimétrica de Agentes\n' +
                 'Área más grande = mejor rendimiento general', fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(output_dir / "06_radar_chart_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = """
        🎯 PLOT: Radar Chart Multimétrico
        
        QUÉ MUESTRA:
        Comparación visual de múltiples métricas en formato radial
        
        INTERPRETACIÓN:
        - Área mayor = mejor rendimiento general
        - Forma del polígono = perfil de fortalezas/debilidades
        - Distancia del centro = nivel de cada métrica
        
        MÉTRICAS INCLUIDAS:
        - Success Rate: % de queries resueltas exitosamente
        - Perfect Rate: % de queries con respuesta perfecta
        - Quality Score: Puntuación promedio de calidad
        - Technical Success: % de ejecuciones técnicamente exitosas
        - Found Rate: % de queries donde se encontró algún resultado
        
        USO EN PRESENTACIÓN:
        - Slide principal para mostrar profiles completos
        - Identificar agentes especialistas vs generalistas
        - Comparar balances entre diferentes aspectos
        """
        
        with open(output_dir / "06_radar_chart_comparison_DOC.txt", "w") as f:
            f.write(doc_text)
    
    def _plot_pareto_frontier(self, agents_data: Dict, output_dir: Path):
        """
        PLOT COMPARATIVO 2: Pareto frontier Quality vs Speed
        """
        agents = list(agents_data.keys())
        quality_scores = []
        efficiency_scores = []
        labels = []
        
        for agent in agents:
            quality = agents_data[agent].get('average_quality_score', 0)
            avg_time = agents_data[agent].get('avg_execution_time', 1)
            efficiency = 1 / avg_time if avg_time > 0 else 0  # Inverso del tiempo = eficiencia
            
            quality_scores.append(quality)
            efficiency_scores.append(efficiency)
            labels.append(agent.replace('agent_', '').title())
        
        # Encontrar puntos en la frontera de Pareto
        pareto_points = []
        for i, (q1, e1) in enumerate(zip(quality_scores, efficiency_scores)):
            is_pareto = True
            for j, (q2, e2) in enumerate(zip(quality_scores, efficiency_scores)):
                if i != j and q2 >= q1 and e2 >= e1 and (q2 > q1 or e2 > e1):
                    is_pareto = False
                    break
            if is_pareto:
                pareto_points.append(i)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot de todos los agentes
        scatter = ax.scatter(efficiency_scores, quality_scores, s=200, alpha=0.7, c=range(len(agents)), cmap='viridis')
        
        # Resaltar puntos de Pareto
        if pareto_points:
            pareto_efficiency = [efficiency_scores[i] for i in pareto_points]
            pareto_quality = [quality_scores[i] for i in pareto_points]
            ax.scatter(pareto_efficiency, pareto_quality, s=300, facecolors='none', 
                      edgecolors='red', linewidths=3, label='Frontera de Pareto')
        
        # Etiquetas
        for i, label in enumerate(labels):
            ax.annotate(label, (efficiency_scores[i], quality_scores[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Eficiencia (1/Tiempo Promedio)', fontweight='bold')
        ax.set_ylabel('Quality Score Promedio', fontweight='bold')
        ax.set_title('⚡ Frontera de Pareto: Calidad vs Eficiencia\n' +
                    'Puntos rojos = Óptimos (no dominados)', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "07_pareto_frontier.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = f"""
        ⚡ PLOT: Frontera de Pareto
        
        QUÉ MUESTRA:
        Trade-off óptimo entre calidad y eficiencia (velocidad)
        
        FRONTERA DE PARETO:
        Agentes marcados en rojo: {[labels[i] for i in pareto_points]}
        
        INTERPRETACIÓN:
        - Arriba-Derecha: IDEAL (alta calidad + alta eficiencia)
        - Puntos en borde rojo: No dominados (óptimos en su trade-off)
        - Puntos internos: Subóptimos (hay otros mejores en ambas métricas)
        
        USO EN PRESENTACIÓN:
        - Identificar agentes con mejor balance
        - Mostrar que no existe "el mejor absoluto"
        - Decisión depende de prioridades (calidad vs velocidad)
        """
        
        with open(output_dir / "07_pareto_frontier_DOC.txt", "w") as f:
            f.write(doc_text)
    
    def generate_error_difficulty_analysis(self):
        """
        Genera análisis específicos de errores y dificultad para cada agente.
        
        Nuevos plots incluyen:
        1. Rendimiento por errores vs consultas limpias
        2. Impacto del nivel de dificultad
        3. Análisis detallado de tipos de errores específicos
        4. Comparación de robustez ante errores por agente
        """
        print("\n🧪 Generando análisis específico de errores y dificultad...")
        
        if not self.raw_data.get('results_by_agent'):
            print("   ⚠️  No hay datos detallados para análisis de errores")
            return
        
        for agent, results in self.raw_data['results_by_agent'].items():
            print(f"   └── Procesando análisis de errores para {agent}...")
            agent_dir = self.plots_dir / agent
            agent_dir.mkdir(exist_ok=True)
            
            # Plot 1: Errores vs Limpias
            self._plot_errors_vs_clean_analysis(agent, results, agent_dir)
            
            # Plot 2: Impacto de dificultad
            self._plot_difficulty_impact_analysis(agent, results, agent_dir)
            
            # Plot 3: Tipos de errores específicos
            self._plot_specific_error_types_analysis(agent, results, agent_dir)
        
        # Plot 4: Comparación de robustez entre agentes
        comparison_dir = self.plots_dir / "comparisons"
        comparison_dir.mkdir(exist_ok=True)
        self._plot_error_robustness_comparison(comparison_dir)
    
    def _plot_errors_vs_clean_analysis(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        PLOT ESPECÍFICO: Análisis de rendimiento con errores vs consultas limpias
        """
        # Separar resultados por errores vs limpias
        clean_results = []
        error_results = []
        
        for r in results:
            dataset_info = r.get('dataset_info', {})
            has_errors = dataset_info.get('has_errors', False)
            
            if has_errors:
                error_results.append(r)
            else:
                clean_results.append(r)
        
        if not clean_results or not error_results:
            print(f"   ⚠️  {agent_name}: No hay suficientes datos para comparar errores vs limpias")
            return
        
        # Reorganizado: Solo 2 plots horizontales
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Success rate comparison + Quality score comparison
        clean_success = sum(1 for r in clean_results if r.get('combined_success', False)) / len(clean_results)
        error_success = sum(1 for r in error_results if r.get('combined_success', False)) / len(error_results)
        
        clean_quality = np.mean([r.get('quality_score', 0) for r in clean_results])
        error_quality = np.mean([r.get('quality_score', 0) for r in error_results])
        
        categories = ['Success Rate', 'Quality Score']
        clean_values = [clean_success, clean_quality]
        error_values = [error_success, error_quality]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, clean_values, width, label='Consultas Limpias', 
                       color='#27ae60', alpha=0.7)
        bars2 = ax1.bar(x + width/2, error_values, width, label='Consultas con Errores', 
                       color='#e74c3c', alpha=0.7)
        
        ax1.set_title('📊 Comparación: Limpias vs Con Errores', fontweight='bold')
        ax1.set_ylabel('Valor de la Métrica')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bars, values in zip([bars1, bars2], [clean_values, error_values]):
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Añadir diferencias
        success_diff = (clean_success - error_success) * 100
        quality_diff = (clean_quality - error_quality) * 100
        ax1.text(0.5, 0.9, f'Diferencias:\nSuccess: {success_diff:+.1f}%\nQuality: {quality_diff:+.1f}%', 
                transform=ax1.transAxes, ha='center', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Plot 2: Distribución de tiers comparativa
        def get_tier_distribution(results_list):
            tiers = {'perfect': 0, 'top_3': 0, 'top_5': 0, 'found_far': 0, 'not_found': 0}
            for r in results_list:
                tier = r.get('scoring_tier', 'not_found')
                if tier in tiers:
                    tiers[tier] += 1
            total = sum(tiers.values())
            return {k: v/total*100 if total > 0 else 0 for k, v in tiers.items()}
        
        clean_tiers = get_tier_distribution(clean_results)
        error_tiers = get_tier_distribution(error_results)
        
        tier_names = list(clean_tiers.keys())
        clean_tier_values = list(clean_tiers.values())
        error_tier_values = list(error_tiers.values())
        
        x_tiers = np.arange(len(tier_names))
        width_tiers = 0.35
        
        bars_clean = ax2.bar(x_tiers - width_tiers/2, clean_tier_values, width_tiers, 
                           label='Limpias', color='#27ae60', alpha=0.7)
        bars_error = ax2.bar(x_tiers + width_tiers/2, error_tier_values, width_tiers, 
                           label='Con Errores', color='#e74c3c', alpha=0.7)
        
        ax2.set_title('📊 Distribución de Tiers: Limpias vs Con Errores', fontweight='bold')
        ax2.set_ylabel('Porcentaje (%)')
        ax2.set_xticks(x_tiers)
        ax2.set_xticklabels([t.replace('_', ' ').title() for t in tier_names], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'🧪 Análisis Errores vs Limpias: {agent_name.replace("agent_", "").title()}\n' +
                    f'Limpias: {len(clean_results)} samples | Con Errores: {len(error_results)} samples', 
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_errors_vs_clean_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_difficulty_impact_analysis(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        PLOT ESPECÍFICO: Análisis del impacto del nivel de dificultad
        """
        # Agrupar por nivel de dificultad
        difficulty_analysis = {}
        
        for r in results:
            dataset_info = r.get('dataset_info', {})
            difficulty = dataset_info.get('difficulty_level', 'unknown')
            
            if difficulty not in difficulty_analysis:
                difficulty_analysis[difficulty] = {
                    'results': [],
                    'success_count': 0,
                    'total_quality': 0,
                    'total_time': 0,
                    'error_count': 0,
                    'clean_count': 0
                }
            
            difficulty_analysis[difficulty]['results'].append(r)
            if r.get('combined_success', False):
                difficulty_analysis[difficulty]['success_count'] += 1
            difficulty_analysis[difficulty]['total_quality'] += r.get('quality_score', 0)
            difficulty_analysis[difficulty]['total_time'] += r.get('execution_time_seconds', 0)
            
            if dataset_info.get('has_errors', False):
                difficulty_analysis[difficulty]['error_count'] += 1
            else:
                difficulty_analysis[difficulty]['clean_count'] += 1
        
        if len(difficulty_analysis) <= 1:
            print(f"   ⚠️  {agent_name}: No hay suficiente variedad de dificultades")
            return
        
        # Reorganizado: Solo 2 plots horizontales
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Ordenar dificultades
        difficulty_order = ['facil', 'medio', 'alto']
        difficulties = [d for d in difficulty_order if d in difficulty_analysis]
        
        # Plot 1: Success rate y Quality score por dificultad
        success_rates = []
        avg_quality = []
        sample_counts = []
        
        for diff in difficulties:
            total = len(difficulty_analysis[diff]['results'])
            success = difficulty_analysis[diff]['success_count']
            quality = difficulty_analysis[diff]['total_quality'] / total if total > 0 else 0
            
            success_rates.append(success / total if total > 0 else 0)
            avg_quality.append(quality)
            sample_counts.append(total)
        
        # Success rate bars
        colors = ['#27ae60', '#f39c12', '#e74c3c'][:len(difficulties)]
        x = np.arange(len(difficulties))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, success_rates, width, label='Success Rate', 
                       color=colors, alpha=0.7)
        
        # Quality score bars (escalado para visualización)
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x + width/2, avg_quality, width, label='Quality Score', 
                            color=colors, alpha=0.5)
        
        ax1.set_title('📊 Success Rate y Quality Score por Dificultad', fontweight='bold')
        ax1.set_ylabel('Success Rate', color='black', fontweight='bold')
        ax1_twin.set_ylabel('Quality Score', color='gray', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([d.title() for d in difficulties])
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bar, rate, count in zip(bars1, success_rates, sample_counts):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{rate:.2%}\n(n={count})', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        for bar, quality in zip(bars2, avg_quality):
            ax1_twin.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                         f'{quality:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 2: Proporción errores vs limpias por dificultad
        error_props = []
        clean_props = []
        
        for diff in difficulties:
            total = len(difficulty_analysis[diff]['results'])
            error_count = difficulty_analysis[diff]['error_count']
            clean_count = difficulty_analysis[diff]['clean_count']
            
            error_props.append(error_count / total * 100 if total > 0 else 0)
            clean_props.append(clean_count / total * 100 if total > 0 else 0)
        
        bars_clean = ax2.bar(x - width/2, clean_props, width, label='Consultas Limpias', 
                           color='#27ae60', alpha=0.7)
        bars_error = ax2.bar(x + width/2, error_props, width, label='Consultas con Errores', 
                           color='#e74c3c', alpha=0.7)
        
        ax2.set_title('📊 Distribución Errores vs Limpias por Dificultad', fontweight='bold')
        ax2.set_ylabel('Porcentaje (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([d.title() for d in difficulties])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bars, values in zip([bars_clean, bars_error], [clean_props, error_props]):
            for bar, value in zip(bars, values):
                if value > 5:  # Solo mostrar si es significativo
                    ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                            f'{value:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        plt.suptitle(f'📈 Análisis de Impacto de Dificultad: {agent_name.replace("agent_", "").title()}\n' +
                    'Cómo afecta la dificultad al rendimiento del agente', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_difficulty_impact_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_specific_error_types_analysis(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        PLOT ESPECÍFICO: Análisis detallado de tipos de errores específicos
        
        CORREGIDO: Elimina el problema del doble conteo calculando correctamente
        el impacto de cada tipo de error de forma independiente.
        """
        # Separar consultas limpias vs con errores
        clean_results = [r for r in results if not r.get('dataset_info', {}).get('has_errors', False)]
        error_results = [r for r in results if r.get('dataset_info', {}).get('has_errors', False)]
        
        if not clean_results or not error_results:
            print(f"   ⚠️  {agent_name}: No hay suficientes datos para analizar tipos de errores")
            return
        
        # Calcular baseline de consultas limpias
        baseline_success = sum(1 for r in clean_results if r.get('combined_success', False)) / len(clean_results)
        baseline_quality = sum(r.get('quality_score', 0) for r in clean_results) / len(clean_results)
        
        print(f"   📊 Baseline limpio para {agent_name}: {baseline_success:.3f} success rate")
        
        # Definir tipos de errores a analizar
        error_categories = {
            'Tildes': ['tilde'],
            'Espaciado': ['espaciado'],
            'Ortográficos': ['ortográf', 'ortograf'],
            'Mayúsculas': ['mayúscula', 'minúscula'],
            'Abreviaciones': ['abreviación', 'abreviacion'],
            'Sustituciones': ['sustitución', 'sustitucion']
        }
        
        # Analizar cada tipo de error independientemente
        error_analysis = {}
        
        for error_name, keywords in error_categories.items():
            # Encontrar consultas que tienen este tipo de error específico
            consultas_con_error = []
            consultas_sin_error = []
            
            for r in error_results:
                error_types = r.get('dataset_info', {}).get('error_types', [])
                tiene_este_error = any(
                    any(keyword in error_type.lower() for keyword in keywords)
                    for error_type in error_types
                )
                
                if tiene_este_error:
                    consultas_con_error.append(r)
                else:
                    consultas_sin_error.append(r)
            
            # Solo analizar si tenemos suficientes datos
            if len(consultas_con_error) < 5:  # Mínimo 5 consultas para ser estadísticamente relevante
                continue
            
            # Calcular métricas para consultas CON este tipo de error
            success_con_error = sum(1 for r in consultas_con_error if r.get('combined_success', False)) / len(consultas_con_error)
            quality_con_error = sum(r.get('quality_score', 0) for r in consultas_con_error) / len(consultas_con_error)
            
            # Calcular métricas para consultas SIN este tipo de error (pero con otros errores)
            success_sin_error = 0
            quality_sin_error = 0
            if consultas_sin_error:
                success_sin_error = sum(1 for r in consultas_sin_error if r.get('combined_success', False)) / len(consultas_sin_error)
                quality_sin_error = sum(r.get('quality_score', 0) for r in consultas_sin_error) / len(consultas_sin_error)
            
            # Calcular impactos
            impacto_vs_baseline = (success_con_error - baseline_success) * 100
            impacto_vs_otros_errores = (success_con_error - success_sin_error) * 100 if consultas_sin_error else 0
            
            error_analysis[error_name] = {
                'count_con_error': len(consultas_con_error),
                'count_sin_error': len(consultas_sin_error),
                'success_con_error': success_con_error,
                'success_sin_error': success_sin_error,
                'quality_con_error': quality_con_error,
                'quality_sin_error': quality_sin_error,
                'impacto_vs_baseline': impacto_vs_baseline,
                'impacto_vs_otros_errores': impacto_vs_otros_errores,
                'consultas_con_error': consultas_con_error,
                'consultas_sin_error': consultas_sin_error
            }
            
            print(f"   └── {error_name}: {len(consultas_con_error)} consultas, impacto vs baseline: {impacto_vs_baseline:+.1f}%")
        
        if not error_analysis:
            print(f"   ⚠️  {agent_name}: No hay suficientes tipos de errores con datos significativos")
            return
        
        # Ordenar por frecuencia
        sorted_errors = sorted(error_analysis.items(), key=lambda x: x[1]['count_con_error'], reverse=True)
        
        # Crear la gráfica SIN EL PIE CHART DE SEVERIDAD
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        error_names = [name for name, _ in sorted_errors]
        colors = plt.cm.Set3(np.linspace(0, 1, len(error_names)))
        
        # Plot 1: Success Rate - Con Error vs Sin Error vs Baseline
        success_con_error = [error_analysis[name]['success_con_error'] for name in error_names]
        success_sin_error = [error_analysis[name]['success_sin_error'] for name in error_names]
        counts_con_error = [error_analysis[name]['count_con_error'] for name in error_names]
        
        x = np.arange(len(error_names))
        width = 0.25
        
        bars1 = ax1.bar(x - width, success_con_error, width, label='Con este error', color='#e74c3c', alpha=0.8)
        bars2 = ax1.bar(x, success_sin_error, width, label='Con otros errores', color='#f39c12', alpha=0.8)
        bars3 = ax1.bar(x + width, [baseline_success] * len(error_names), width, label='Baseline limpio', color='#27ae60', alpha=0.8)
        
        ax1.set_title('📊 Success Rate: Impacto Específico por Tipo de Error', fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_xticks(x)
        ax1.set_xticklabels(error_names, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)
        
        # Añadir conteos en las barras
        for bar, count in zip(bars1, counts_con_error):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'n={count}', ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        # Plot 2: Impacto vs Baseline (Degradación)
        impactos_baseline = [error_analysis[name]['impacto_vs_baseline'] for name in error_names]
        
        # Colores por impacto (rojo = negativo, verde = positivo)
        impact_colors = ['#e74c3c' if imp < 0 else '#27ae60' for imp in impactos_baseline]
        
        bars4 = ax2.bar(error_names, impactos_baseline, color=impact_colors, alpha=0.8)
        ax2.set_title(f'📊 Degradación vs Baseline Limpio ({baseline_success:.1%})', fontweight='bold')
        ax2.set_ylabel('Diferencia en Success Rate (%)')
        ax2.tick_params(axis='x', rotation=45)
        ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax2.grid(True, alpha=0.3)
        
        for bar, imp in zip(bars4, impactos_baseline):
            ax2.text(bar.get_x() + bar.get_width()/2., 
                    bar.get_height() + (1 if imp >= 0 else -2),
                    f'{imp:+.1f}%', ha='center', 
                    va='bottom' if imp >= 0 else 'top', 
                    fontsize=9, fontweight='bold')
        
        # Plot 3: Quality Score por tipo de error
        quality_con_error = [error_analysis[name]['quality_con_error'] for name in error_names]
        quality_sin_error = [error_analysis[name]['quality_sin_error'] for name in error_names]
        
        bars5 = ax3.bar(x - width/2, quality_con_error, width, label='Con este error', color='#e74c3c', alpha=0.8)
        bars6 = ax3.bar(x + width/2, quality_sin_error, width, label='Con otros errores', color='#f39c12', alpha=0.8)
        
        ax3.axhline(y=baseline_quality, color='#27ae60', linestyle='--', alpha=0.8, linewidth=2, label=f'Baseline limpio ({baseline_quality:.3f})')
        ax3.set_title('📊 Quality Score por Tipo de Error', fontweight='bold')
        ax3.set_ylabel('Quality Score Promedio')
        ax3.set_xticks(x)
        ax3.set_xticklabels(error_names, rotation=45, ha='right')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'🧪 Análisis de Tipos de Errores Específicos: {agent_name.replace("agent_", "").title()}\n' +
                    f'Análisis independiente por tipo de error sin doble conteo', 
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_specific_error_types_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_error_robustness_comparison(self, output_dir: Path):
        """
        PLOT COMPARATIVO: Comparación de robustez ante errores entre agentes
        """
        if not self.raw_data.get('results_by_agent'):
            return
        
        agents_robustness = {}
        
        for agent, results in self.raw_data['results_by_agent'].items():
            clean_results = [r for r in results if not r.get('dataset_info', {}).get('has_errors', False)]
            error_results = [r for r in results if r.get('dataset_info', {}).get('has_errors', False)]
            
            if not clean_results or not error_results:
                continue
            
            clean_success = sum(1 for r in clean_results if r.get('combined_success', False)) / len(clean_results)
            error_success = sum(1 for r in error_results if r.get('combined_success', False)) / len(error_results)
            
            clean_quality = sum(r.get('quality_score', 0) for r in clean_results) / len(clean_results)
            error_quality = sum(r.get('quality_score', 0) for r in error_results) / len(error_results)
            
            agents_robustness[agent] = {
                'clean_success': clean_success,
                'error_success': error_success,
                'success_degradation': (clean_success - error_success) * 100,
                'clean_quality': clean_quality,
                'error_quality': error_quality,
                'quality_degradation': (clean_quality - error_quality) * 100,
                'robustness_score': error_success / clean_success if clean_success > 0 else 0
            }
        
        if len(agents_robustness) <= 1:
            print("   ⚠️  No hay suficientes agentes para comparar robustez")
            return
        
        # Reorganizado: Solo 2 plots horizontales
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        agents = list(agents_robustness.keys())
        agent_labels = [agent.replace('agent_', '').title() for agent in agents]
        
        # Plot 1: Success rate limpias vs errores
        clean_rates = [agents_robustness[agent]['clean_success'] for agent in agents]
        error_rates = [agents_robustness[agent]['error_success'] for agent in agents]
        
        x = np.arange(len(agents))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, clean_rates, width, label='Consultas Limpias', 
                       color='#27ae60', alpha=0.7)
        bars2 = ax1.bar(x + width/2, error_rates, width, label='Consultas con Errores', 
                       color='#e74c3c', alpha=0.7)
        
        ax1.set_title('📊 Comparación Success Rate: Limpias vs Errores', fontweight='bold')
        ax1.set_ylabel('Success Rate')
        ax1.set_xticks(x)
        ax1.set_xticklabels(agent_labels, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Añadir valores en las barras
        for bars, rates in zip([bars1, bars2], [clean_rates, error_rates]):
            for bar, rate in zip(bars, rates):
                ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                        f'{rate:.2%}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 2: Ranking de robustez con degradación
        robustness_scores = [agents_robustness[agent]['robustness_score'] for agent in agents]
        degradations = [agents_robustness[agent]['success_degradation'] for agent in agents]
        
        # Ordenar por robustez
        sorted_data = sorted(zip(agent_labels, robustness_scores, degradations), key=lambda x: x[1], reverse=True)
        sorted_labels, sorted_scores, sorted_degradations = zip(*sorted_data)
        
        # Barras de robustez
        colors_rank = plt.cm.RdYlGn(np.array(sorted_scores))
        bars_rob = ax2.bar(sorted_labels, sorted_scores, color=colors_rank, alpha=0.7, label='Score de Robustez')
        
        # Línea de degradación en eje secundario
        ax2_twin = ax2.twinx()
        line_deg = ax2_twin.plot(sorted_labels, sorted_degradations, 'ro-', linewidth=2, markersize=8, 
                                label='Degradación (%)', color='red')
        
        ax2.set_title('🏆 Ranking de Robustez y Degradación', fontweight='bold')
        ax2.set_ylabel('Score de Robustez (Error/Limpio)', color='black', fontweight='bold')
        ax2_twin.set_ylabel('Degradación (%)', color='red', fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1.1)
        
        # Añadir valores en las barras
        for bar, score in zip(bars_rob, sorted_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Añadir valores en la línea
        for i, (label, deg) in enumerate(zip(sorted_labels, sorted_degradations)):
            ax2_twin.text(i, deg + 1, f'{deg:+.1f}%', ha='center', va='bottom', 
                         fontweight='bold', fontsize=9, color='red')
        
        # Añadir línea de referencia
        ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Robustez perfecta (1.0)')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        plt.suptitle('🛡️ Análisis Comparativo de Robustez ante Errores\n' +
                    'Qué agentes manejan mejor las consultas con errores inyectados', 
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "10_error_robustness_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """
    Función principal para ejecutar todo el análisis.
    
    Uso: python generate_analysis_plots.py [archivo_resultados.json]
    """
    print("🚀 Iniciando análisis completo de agentes...")
    
    # Verificar argumentos de línea de comandos
    results_file = None
    if len(sys.argv) > 1:
        results_file = sys.argv[1]
        print(f"📂 Archivo especificado: {results_file}")
    else:
        print("📂 No se especificó archivo, buscando el más reciente...")
    
    plotter = AgentAnalysisPlotter(results_file)
    
    # Cargar datos
    if not plotter.load_data():
        print("❌ Error: No se pudieron cargar los datos")
        print("\n💡 Uso correcto:")
        print("   python generate_analysis_plots.py")
        print("   python generate_analysis_plots.py results_local_10samples_phi4_14b_phi4_reasoning_14b_20250602_150120.json")
        return
    
    # Generar plots
    plotter.generate_comparison_plots()
    plotter.generate_individual_plots()
    plotter.create_summary_report()
    
    # Generar plots específicos avanzados
    plotter.generate_advanced_specific_plots()
    
    # Generar plots comparativos avanzados
    plotter.generate_advanced_comparison_plots()
    
    # 🆕 NUEVO: Generar análisis específico de errores y dificultad
    plotter.generate_error_difficulty_analysis()
    
    print("\n✅ Análisis completo finalizado!")
    print("📁 Revisa la carpeta /plots/ para todos los gráficos y documentación")
    print("📋 Lee /plots/EXECUTIVE_SUMMARY.txt para insights principales")
    print(f"\n📊 Datos procesados desde: {plotter.results_file}")
    print("\n🆕 NUEVAS GRÁFICAS AÑADIDAS:")
    print("   🧪 Análisis errores vs consultas limpias por agente")
    print("   📈 Impacto del nivel de dificultad por agente") 
    print("   🔍 Análisis de tipos de errores específicos por agente")
    print("   🛡️ Comparación de robustez ante errores entre agentes")

if __name__ == "__main__":
    main() 
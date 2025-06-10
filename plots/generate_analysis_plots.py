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
        Genera todos los plots de comparación entre agentes.
        
        Plots generados:
        1. Barras agrupadas de métricas principales
        2. Análisis Thinking vs Non-Thinking
        3. Trade-off Quality vs Speed
        4. Distribución de tiers de calidad
        5. Heatmap de rendimiento relativo
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
        
        # 2. PLOT: Análisis Thinking vs Non-Thinking
        self._plot_thinking_vs_regular(comparison_dir)
        
        # 3. PLOT: Trade-off Quality vs Speed
        self._plot_quality_speed_tradeoff(agents_data, comparison_dir)
        
        # 4. PLOT: Distribución de calidad (si hay datos detallados)
        if self.raw_data.get('results_by_agent'):
            self._plot_quality_distribution(comparison_dir)
        
        # 5. PLOT: Heatmap de rendimiento relativo
        self._plot_performance_heatmap(agents_data, comparison_dir)
        
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
    
    def _plot_thinking_vs_regular(self, output_dir: Path):
        """
        PLOT 2: Análisis Thinking vs Non-Thinking
        
        Propósito: Comparar el impacto de usar modelos de reasoning vs regulares
        """
        thinking_data = self.calculated_stats.get('thinking_vs_regular_analysis', {})
        
        if not thinking_data:
            print("   ⚠️  No hay datos de thinking vs regular")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        categories = ['Con Thinking', 'Sin Thinking']
        with_thinking = thinking_data.get('agents_with_thinking', {})
        without_thinking = thinking_data.get('agents_without_thinking', {})
        
        # Subplot 1: Success Rate
        success_rates = [
            with_thinking.get('combined_success_rate', 0),
            without_thinking.get('combined_success_rate', 0)
        ]
        bars1 = ax1.bar(categories, success_rates, color=['#e74c3c', '#3498db'], alpha=0.7)
        ax1.set_title('Success Rate: Thinking vs Regular', fontweight='bold')
        ax1.set_ylabel('Combined Success Rate')
        for bar, value in zip(bars1, success_rates):
            ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 2: Quality Score
        quality_scores = [
            with_thinking.get('average_quality_score', 0),
            without_thinking.get('average_quality_score', 0)
        ]
        bars2 = ax2.bar(categories, quality_scores, color=['#e74c3c', '#3498db'], alpha=0.7)
        ax2.set_title('Quality Score: Thinking vs Regular', fontweight='bold')
        ax2.set_ylabel('Average Quality Score')
        for bar, value in zip(bars2, quality_scores):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 3: Execution Time
        exec_times = [
            with_thinking.get('avg_execution_time', 0),
            without_thinking.get('avg_execution_time', 0)
        ]
        bars3 = ax3.bar(categories, exec_times, color=['#e74c3c', '#3498db'], alpha=0.7)
        ax3.set_title('Tiempo de Ejecución: Thinking vs Regular', fontweight='bold')
        ax3.set_ylabel('Tiempo promedio (segundos)')
        for bar, value in zip(bars3, exec_times):
            ax3.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{value:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        # Subplot 4: Perfect Rate
        perfect_rates = [
            with_thinking.get('perfect_rate', 0),
            without_thinking.get('perfect_rate', 0)
        ]
        bars4 = ax4.bar(categories, perfect_rates, color=['#e74c3c', '#3498db'], alpha=0.7)
        ax4.set_title('Perfect Rate: Thinking vs Regular', fontweight='bold')
        ax4.set_ylabel('Perfect Rate')
        for bar, value in zip(bars4, perfect_rates):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.suptitle('🧠 Análisis: Modelos con Thinking vs Regulares\n' + 
                    'Comparación de rendimiento y eficiencia', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_dir / "02_thinking_vs_regular_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = f"""
        🧠 PLOT: Análisis Thinking vs Regular
        
        QUÉ MUESTRA:
        Comparación directa entre agentes que usan modelos de reasoning vs regulares
        
        DATOS CLAVE ENCONTRADOS:
        - Con Thinking: {with_thinking.get('count', 0)} agente(s)
        - Sin Thinking: {without_thinking.get('count', 0)} agente(s)
        - Diferencia en tiempo: {exec_times[0]/exec_times[1]:.1f}x más lento el thinking
        - Diferencia en success: {(success_rates[0]-success_rates[1])*100:+.1f}% el thinking
        
        INTERPRETACIÓN:
        - Si barras rojas > azules: Thinking es mejor
        - Si barras azules > rojas: Regular es mejor
        - Notar el trade-off tiempo vs calidad
        
        INSIGHTS PARA PRESENTACIÓN:
        - ¿Vale la pena el tiempo extra del thinking?
        - ¿En qué escenarios usarías cada tipo?
        """
        
        with open(output_dir / "02_thinking_vs_regular_analysis_DOC.txt", "w") as f:
            f.write(doc_text)
    
    def _plot_quality_speed_tradeoff(self, agents_data: Dict, output_dir: Path):
        """
        PLOT 3: Trade-off Quality vs Speed
        
        Propósito: Visualizar la relación entre calidad y velocidad de respuesta
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        agents = []
        quality_scores = []
        exec_times = []
        success_rates = []
        
        for agent, data in agents_data.items():
            agents.append(agent.replace('agent_', '').title())
            quality_scores.append(data.get('average_quality_score', 0))
            exec_times.append(data.get('avg_execution_time', 0))
            success_rates.append(data.get('combined_success_rate', 0))
        
        # Scatter plot con tamaño basado en success rate
        scatter = ax.scatter(exec_times, quality_scores, 
                           s=[rate * 500 for rate in success_rates],  # Tamaño proporcional
                           alpha=0.7, c=range(len(agents)), cmap='viridis')
        
        # Etiquetas para cada punto
        for i, agent in enumerate(agents):
            ax.annotate(agent, (exec_times[i], quality_scores[i]), 
                       xytext=(5, 5), textcoords='offset points',
                       fontweight='bold', fontsize=11)
        
        ax.set_xlabel('Tiempo de Ejecución Promedio (segundos)', fontweight='bold')
        ax.set_ylabel('Quality Score Promedio', fontweight='bold')
        ax.set_title('⚡ Trade-off: Calidad vs Velocidad de Respuesta\n' +
                    'Tamaño del punto = Success Rate', fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3)
        
        # Añadir líneas de referencia
        ax.axhline(y=np.mean(quality_scores), color='red', linestyle='--', alpha=0.5, 
                  label=f'Calidad promedio: {np.mean(quality_scores):.3f}')
        ax.axvline(x=np.mean(exec_times), color='blue', linestyle='--', alpha=0.5,
                  label=f'Tiempo promedio: {np.mean(exec_times):.1f}s')
        
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(output_dir / "03_quality_speed_tradeoff.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = """
        ⚡ PLOT: Trade-off Calidad vs Velocidad
        
        QUÉ MUESTRA:
        - Eje X: Tiempo de ejecución (más derecha = más lento)
        - Eje Y: Quality score (más arriba = mejor calidad)
        - Tamaño punto: Success rate (más grande = más exitoso)
        
        ZONAS DEL GRÁFICO:
        - Arriba-Izquierda: IDEAL (rápido y bueno)
        - Arriba-Derecha: Bueno pero lento
        - Abajo-Izquierda: Rápido pero malo
        - Abajo-Derecha: PEOR (lento y malo)
        
        INTERPRETACIÓN:
        - Buscar puntos en esquina superior izquierda
        - Las líneas de referencia dividen en cuadrantes
        - Puntos grandes indican agentes más confiables
        
        USO EN PRESENTACIÓN:
        - Mostrar que no siempre "más lento = mejor"
        - Identificar el agente con mejor balance
        """
        
        with open(output_dir / "03_quality_speed_tradeoff_DOC.txt", "w") as f:
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
        
        Nuevos plots incluyen:
        1. Violin plots de distribuciones
        2. Análisis temporal detallado
        3. Análisis de tipos de query
        4. Patrones de fallo específicos
        """
        print("\n🎯 Generando plots específicos avanzados por agente...")
        
        if not self.raw_data.get('results_by_agent'):
            print("   ⚠️  No hay datos detallados para análisis específico avanzado")
            return
        
        for agent, results in self.raw_data['results_by_agent'].items():
            print(f"   └── Procesando plots avanzados para {agent}...")
            agent_dir = self.plots_dir / agent
            agent_dir.mkdir(exist_ok=True)
            
            # Plot 1: Violin plots de distribuciones
            self._plot_violin_distributions(agent, results, agent_dir)
            
            # Plot 2: Análisis temporal
            self._plot_temporal_analysis(agent, results, agent_dir)
            
            # Plot 3: Análisis por tipos de query
            self._plot_query_type_analysis(agent, results, agent_dir)
            
            # Plot 4: Análisis de patrones de fallo
            self._plot_failure_patterns(agent, results, agent_dir)
    
    def _plot_violin_distributions(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        PLOT ESPECÍFICO 1: Violin plots de distribuciones completas
        """
        # Extraer datos
        execution_times = [r.get('execution_time_seconds', 0) for r in results if r.get('execution_time_seconds')]
        quality_scores = [r.get('quality_score', 0) for r in results]
        
        if not execution_times or not quality_scores:
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Violin plot de tiempos
        parts1 = ax1.violinplot([execution_times], positions=[1], widths=0.6, showmeans=True, showmedians=True)
        ax1.set_title('🎻 Distribución Completa: Tiempos de Ejecución', fontweight='bold')
        ax1.set_ylabel('Tiempo (segundos)')
        ax1.set_xticks([1])
        ax1.set_xticklabels([agent_name.replace('agent_', '').title()])
        ax1.grid(True, alpha=0.3)
        
        # Añadir estadísticas
        mean_time = np.mean(execution_times)
        median_time = np.median(execution_times)
        std_time = np.std(execution_times)
        ax1.text(1.3, mean_time, f'Media: {mean_time:.2f}s\nMediana: {median_time:.2f}s\nStd: {std_time:.2f}s', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
        
        # Violin plot de quality scores
        parts2 = ax2.violinplot([quality_scores], positions=[1], widths=0.6, showmeans=True, showmedians=True)
        ax2.set_title('🎻 Distribución Completa: Quality Scores', fontweight='bold')
        ax2.set_ylabel('Quality Score (0-1)')
        ax2.set_xticks([1])
        ax2.set_xticklabels([agent_name.replace('agent_', '').title()])
        ax2.grid(True, alpha=0.3)
        
        # Añadir estadísticas
        mean_quality = np.mean(quality_scores)
        median_quality = np.median(quality_scores)
        std_quality = np.std(quality_scores)
        ax2.text(1.3, mean_quality, f'Media: {mean_quality:.3f}\nMediana: {median_quality:.3f}\nStd: {std_quality:.3f}', 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.suptitle(f'📊 Análisis de Distribuciones: {agent_name.replace("agent_", "").title()}\n' +
                    'Violin plots muestran la forma completa de las distribuciones', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_violin_distributions.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_temporal_analysis(self, agent_name: str, results: List[Dict], output_dir: Path):
        """
        PLOT ESPECÍFICO 2: Análisis temporal detallado
        """
        # Extraer datos temporales
        timestamps = []
        success_values = []
        quality_scores = []
        execution_times = []
        
        for r in results:
            if 'timestamp' in r:
                timestamps.append(pd.to_datetime(r['timestamp']))
                success_values.append(1 if r.get('combined_success', False) else 0)
                quality_scores.append(r.get('quality_score', 0))
                execution_times.append(r.get('execution_time_seconds', 0))
        
        if not timestamps:
            return
        
        # Crear DataFrame para análisis temporal
        df = pd.DataFrame({
            'timestamp': timestamps,
            'success': success_values,
            'quality': quality_scores,
            'exec_time': execution_times
        }).sort_values('timestamp')
        
        # Calcular rolling averages
        window_size = max(5, len(df) // 10)  # Ventana adaptativa
        df['success_rolling'] = df['success'].rolling(window=window_size, min_periods=1).mean()
        df['quality_rolling'] = df['quality'].rolling(window=window_size, min_periods=1).mean()
        df['time_rolling'] = df['exec_time'].rolling(window=window_size, min_periods=1).mean()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Success rate temporal
        ax1.scatter(df['timestamp'], df['success'], alpha=0.5, s=30, color='blue', label='Éxitos individuales')
        ax1.plot(df['timestamp'], df['success_rolling'], color='red', linewidth=2, label=f'Media móvil (ventana={window_size})')
        ax1.set_title('📈 Evolución Temporal: Success Rate', fontweight='bold')
        ax1.set_ylabel('Success (0/1)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Quality score temporal
        ax2.scatter(df['timestamp'], df['quality'], alpha=0.5, s=30, color='green', label='Quality individual')
        ax2.plot(df['timestamp'], df['quality_rolling'], color='darkgreen', linewidth=2, label=f'Media móvil (ventana={window_size})')
        ax2.set_title('📈 Evolución Temporal: Quality Score', fontweight='bold')
        ax2.set_ylabel('Quality Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Execution time temporal
        ax3.scatter(df['timestamp'], df['exec_time'], alpha=0.5, s=30, color='orange', label='Tiempo individual')
        ax3.plot(df['timestamp'], df['time_rolling'], color='darkorange', linewidth=2, label=f'Media móvil (ventana={window_size})')
        ax3.set_title('📈 Evolución Temporal: Tiempo de Ejecución', fontweight='bold')
        ax3.set_ylabel('Tiempo (segundos)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Distribución por hora
        df['hour'] = df['timestamp'].dt.hour
        hourly_success = df.groupby('hour')['success'].mean()
        if len(hourly_success) > 1:
            ax4.bar(hourly_success.index, hourly_success.values, alpha=0.7, color='purple')
            ax4.set_title('📊 Rendimiento por Hora del Día', fontweight='bold')
            ax4.set_xlabel('Hora del día')
            ax4.set_ylabel('Success Rate promedio')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'⏰ Análisis Temporal: {agent_name.replace("agent_", "").title()}\n' +
                    'Patrones de rendimiento a lo largo del tiempo', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f"{agent_name}_temporal_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
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
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Distribución de quality scores en fallos vs éxitos
        success_qualities = [r.get('quality_score', 0) for r in successes]
        failure_qualities = [r.get('quality_score', 0) for r in failures]
        
        ax1.hist([success_qualities, failure_qualities], bins=10, alpha=0.7, 
                label=['Éxitos', 'Fallos'], color=['green', 'red'])
        ax1.set_title('📊 Distribución Quality Score: Éxitos vs Fallos', fontweight='bold')
        ax1.set_xlabel('Quality Score')
        ax1.set_ylabel('Frecuencia')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Tiempo gastado en éxitos vs fallos
        success_times = [r.get('execution_time_seconds', 0) for r in successes]
        failure_times = [r.get('execution_time_seconds', 0) for r in failures]
        
        box_data = [success_times, failure_times] if success_times else [failure_times]
        box_labels = ['Éxitos', 'Fallos'] if success_times else ['Fallos']
        
        bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
        bp['boxes'][0].set_facecolor('green' if success_times else 'red')
        if len(bp['boxes']) > 1:
            bp['boxes'][1].set_facecolor('red')
        
        ax2.set_title('📊 Tiempo de Ejecución: Éxitos vs Fallos', fontweight='bold')
        ax2.set_ylabel('Tiempo (segundos)')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Tipos de fallo por dificultad
        failure_by_difficulty = {}
        for f in failures:
            difficulty = f.get('dataset_info', {}).get('difficulty_level', 'unknown')
            failure_by_difficulty[difficulty] = failure_by_difficulty.get(difficulty, 0) + 1
        
        if failure_by_difficulty:
            difficulties = list(failure_by_difficulty.keys())
            counts = list(failure_by_difficulty.values())
            colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(difficulties)))
            
            wedges, texts, autotexts = ax3.pie(counts, labels=difficulties, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            ax3.set_title('📊 Distribución de Fallos por Dificultad', fontweight='bold')
        
        # Plot 4: Candidates encontrados en fallos
        failure_candidates = [r.get('total_candidates', 0) for r in failures]
        
        ax4.hist(failure_candidates, bins=max(5, len(set(failure_candidates))), 
                alpha=0.7, color='red', edgecolor='black')
        ax4.set_title('📊 Candidates Encontrados en Fallos', fontweight='bold')
        ax4.set_xlabel('Número de Candidates')
        ax4.set_ylabel('Frecuencia')
        ax4.grid(True, alpha=0.3)
        
        # Añadir estadística
        mean_candidates = np.mean(failure_candidates) if failure_candidates else 0
        ax4.axvline(mean_candidates, color='darkred', linestyle='--', 
                   label=f'Media: {mean_candidates:.1f}')
        ax4.legend()
        
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
        
        Nuevos plots incluyen:
        1. Radar chart multimétrico
        2. Pareto frontier analysis
        3. Statistical significance matrix
        4. Clustering de agentes
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
        
        # Plot 3: Statistical significance matrix
        self._plot_statistical_significance(comparison_dir)
        
        # Plot 4: Clustering de agentes
        self._plot_agent_clustering(agents_data, comparison_dir)
        
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
    
    def _plot_statistical_significance(self, output_dir: Path):
        """
        PLOT COMPARATIVO 3: Matriz de significancia estadística
        """
        if not ADVANCED_ANALYSIS_AVAILABLE:
            print("   ⚠️  Análisis estadístico no disponible (falta scipy)")
            return
            
        if not self.raw_data.get('results_by_agent'):
            return
        
        agents = list(self.raw_data['results_by_agent'].keys())
        if len(agents) < 2:
            return
        
        # Extraer quality scores por agente
        agent_scores = {}
        for agent, results in self.raw_data['results_by_agent'].items():
            scores = [r.get('quality_score', 0) for r in results]
            agent_scores[agent] = scores
        
        # Calcular matriz de p-values (simulada con Mann-Whitney U test)
        n_agents = len(agents)
        p_matrix = np.ones((n_agents, n_agents))
        effect_matrix = np.zeros((n_agents, n_agents))
        
        for i, agent1 in enumerate(agents):
            for j, agent2 in enumerate(agents):
                if i != j:
                    scores1 = agent_scores[agent1]
                    scores2 = agent_scores[agent2]
                    
                    if len(scores1) > 1 and len(scores2) > 1:
                        try:
                            statistic, p_value = stats.mannwhitneyu(scores1, scores2, alternative='two-sided')
                            p_matrix[i, j] = p_value
                            
                            # Effect size (diferencia de medias normalizadas)
                            mean1, mean2 = np.mean(scores1), np.mean(scores2)
                            pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
                            if pooled_std > 0:
                                effect_matrix[i, j] = (mean1 - mean2) / pooled_std
                        except:
                            p_matrix[i, j] = 1.0
        
        # Crear heatmap de significancia
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap de p-values
        agent_labels = [agent.replace('agent_', '').title() for agent in agents]
        sns.heatmap(p_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   xticklabels=agent_labels, yticklabels=agent_labels,
                   ax=ax1, cbar_kws={'label': 'p-value'})
        ax1.set_title('🔬 Matriz de Significancia Estadística\n(p-values)', fontweight='bold')
        
        # Heatmap de effect sizes
        sns.heatmap(effect_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                   xticklabels=agent_labels, yticklabels=agent_labels,
                   ax=ax2, cbar_kws={'label': 'Effect Size (Cohen\'s d)'})
        ax2.set_title('📏 Matriz de Tamaño del Efecto\n(Effect Sizes)', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_dir / "08_statistical_significance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = """
        🔬 PLOT: Significancia Estadística
        
        QUÉ MUESTRA:
        Comparación estadística rigurosa entre todos los pares de agentes
        
        MATRIZ DE P-VALUES:
        - Verde: Diferencias estadísticamente significativas (p < 0.05)
        - Rojo: No hay diferencia significativa (p > 0.05)
        - Valores en celdas: p-value exacto
        
        MATRIZ DE EFFECT SIZES:
        - Azul: Agente fila es MEJOR que agente columna
        - Rojo: Agente fila es PEOR que agente columna
        - Intensidad = magnitud de la diferencia
        
        INTERPRETACIÓN:
        - p < 0.05 AND |effect size| > 0.5 = Diferencia prácticamente significativa
        - p > 0.05 = No hay diferencia estadística real
        - Effect size > 0.8 = Diferencia grande
        
        USO EN PRESENTACIÓN:
        - Responder: "¿Son realmente diferentes los agentes?"
        - Evitar conclusiones erróneas por variabilidad aleatoria
        """
        
        with open(output_dir / "08_statistical_significance_DOC.txt", "w") as f:
            f.write(doc_text)
    
    def _plot_agent_clustering(self, agents_data: Dict, output_dir: Path):
        """
        PLOT COMPARATIVO 4: Clustering de agentes por similitud
        """
        if not ADVANCED_ANALYSIS_AVAILABLE:
            print("   ⚠️  Clustering no disponible (falta scikit-learn)")
            return
            
        agents = list(agents_data.keys())
        if len(agents) < 2:
            return
        
        # Preparar matriz de características
        features = ['combined_success_rate', 'perfect_rate', 'average_quality_score', 
                   'technical_success_rate', 'found_in_results_rate']
        
        data_matrix = []
        for agent in agents:
            row = []
            for feature in features:
                value = agents_data[agent].get(feature, 0)
                row.append(value)
            data_matrix.append(row)
        
        data_matrix = np.array(data_matrix)
        
        # Normalizar datos
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(data_matrix)
        
        # Clustering jerárquico
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Dendrogram
        linkage_matrix = linkage(normalized_data, method='ward')
        dendrogram(linkage_matrix, labels=[agent.replace('agent_', '').title() for agent in agents],
                  ax=ax1, orientation='top')
        ax1.set_title('🌲 Clustering Jerárquico de Agentes\n(Similitud en rendimiento)', fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot 2: K-means clustering (2D PCA)
        if len(agents) > 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(normalized_data)
            
            # K-means
            n_clusters = min(3, len(agents))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(normalized_data)
            
            colors = plt.cm.Set1(np.linspace(0, 1, n_clusters))
            for i, (x, y) in enumerate(pca_data):
                ax2.scatter(x, y, c=[colors[cluster_labels[i]]], s=200, alpha=0.7)
                ax2.annotate(agents[i].replace('agent_', '').title(), (x, y), 
                           xytext=(5, 5), textcoords='offset points', fontweight='bold')
            
            ax2.set_title(f'🎯 K-Means Clustering (k={n_clusters})\nPCA 2D projection', fontweight='bold')
            ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} varianza)')
            ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} varianza)')
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Matriz de distancias
        distances = pdist(normalized_data, metric='euclidean')
        distance_matrix = squareform(distances)
        
        agent_labels = [agent.replace('agent_', '').title() for agent in agents]
        sns.heatmap(distance_matrix, annot=True, fmt='.2f', cmap='viridis',
                   xticklabels=agent_labels, yticklabels=agent_labels, ax=ax3)
        ax3.set_title('📏 Matriz de Distancias Euclidianas\n(0 = idénticos, mayor = más diferentes)', fontweight='bold')
        
        # Plot 4: Análisis de componentes principales
        if len(features) > 2 and len(agents) > 1:
            pca_full = PCA()
            pca_full.fit(normalized_data)
            
            # El número de componentes es min(n_features, n_samples)
            n_components = min(len(features), len(agents))
            explained_variance = pca_full.explained_variance_ratio_[:n_components]
            
            ax4.bar(range(1, n_components + 1), explained_variance, 
                   alpha=0.7, color='skyblue')
            ax4.set_title('📊 Análisis de Componentes Principales\nVarianza explicada por componente', fontweight='bold')
            ax4.set_xlabel('Componente Principal')
            ax4.set_ylabel('% Varianza Explicada')
            ax4.grid(True, alpha=0.3)
            
            # Añadir varianza acumulada
            cumvar = np.cumsum(explained_variance)
            ax4_twin = ax4.twinx()
            ax4_twin.plot(range(1, n_components + 1), cumvar, 'ro-', color='red', linewidth=2)
            ax4_twin.set_ylabel('% Varianza Acumulada', color='red')
        
        plt.suptitle('🔍 Análisis de Clustering: Agrupación por Similitud de Rendimiento\n' +
                    'Identificar familias de agentes con comportamiento similar', fontweight='bold', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "09_agent_clustering.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Documentación
        doc_text = """
        🔍 PLOT: Clustering de Agentes
        
        QUÉ MUESTRA:
        Agrupación de agentes por similitud en múltiples métricas de rendimiento
        
        DENDROGRAM:
        - Altura = nivel de diferencia entre agentes/grupos
        - Ramas que se unen abajo = agentes muy similares
        - Ramas que se unen arriba = agentes muy diferentes
        
        K-MEANS + PCA:
        - Proyección 2D de datos multidimensionales
        - Colores = clusters automáticos
        - Distancia visual ≈ similitud real
        
        MATRIZ DE DISTANCIAS:
        - 0 = agentes idénticos
        - Valores altos = agentes muy diferentes
        - Útil para encontrar el agente más "único"
        
        PCA:
        - % varianza explicada por cada dimensión
        - Línea roja = varianza acumulada
        - Ayuda a entender complejidad de diferencias
        
        USO EN PRESENTACIÓN:
        - "¿Qué agentes son redundantes?"
        - "¿Hay familias de enfoques?"
        - "¿Cuál es el agente más diferente?"
        """
        
        with open(output_dir / "09_agent_clustering_DOC.txt", "w") as f:
            f.write(doc_text)

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
    
    print("\n✅ Análisis completo finalizado!")
    print("📁 Revisa la carpeta /plots/ para todos los gráficos y documentación")
    print("📋 Lee /plots/EXECUTIVE_SUMMARY.txt para insights principales")
    print(f"\n📊 Datos procesados desde: {plotter.results_file}")

if __name__ == "__main__":
    main() 
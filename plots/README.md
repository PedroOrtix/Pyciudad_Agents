# 📊 Sistema de Análisis Visual de Agentes

Este directorio contiene el sistema completo de análisis gráfico para evaluar y comparar el rendimiento de diferentes agentes. 

**✨ NUEVA FUNCIONALIDAD**: El sistema ahora calcula **TODAS las estadísticas directamente desde archivos JSON específicos** que tú especifiques, sin depender de performance reports.

## 🚀 Uso Rápido

```bash
# Instalar dependencias (desde la raíz del proyecto)
pip install -r ../requirements.txt

# Analizar archivo específico (RECOMENDADO)
python generate_analysis_plots.py results_local_10samples_phi4_14b_phi4_reasoning_14b_20250602_150120.json

# Auto-detectar archivo más reciente
python generate_analysis_plots.py

# Script interactivo con ejemplos
./run_analysis_examples.sh
```

## 🎯 Especificar Archivo de Datos

### Método 1: Archivo Específico
```bash
python generate_analysis_plots.py [nombre_archivo.json]
```

**Ejemplos:**
- `python generate_analysis_plots.py results_local_10samples_phi4_14b_phi4_reasoning_14b_20250602_150120.json`
- `python generate_analysis_plots.py results_local_1500samples_phi4_14b_phi4_reasoning_14b_20250603_081958.json`

### Método 2: Auto-detección
```bash
python generate_analysis_plots.py
```
Busca automáticamente el archivo más reciente en `evaluation/results/`

## 📊 Cálculo de Estadísticas

El sistema ahora:
- ✅ **Lee DIRECTAMENTE** el archivo JSON de resultados que especifiques
- ✅ **Calcula TODAS las estadísticas** desde cero procesando datos individuales
- ✅ **NO depende** de performance reports pre-calculados
- ✅ **Detecta automáticamente** thinking vs regular models
- ✅ **Procesa cada query individual** para máxima precisión

### Estadísticas Calculadas:
- `combined_success_rate`: % de queries resueltas exitosamente
- `perfect_rate`: % de queries con respuesta perfecta (posición 1)
- `top_3_rate` / `top_5_rate`: % de queries en top 3/5
- `average_quality_score`: Puntuación promedio (0-1)
- `avg_execution_time`: Tiempo promedio de ejecución
- `tier_distribution`: Distribución de calidad (perfect/top_3/top_5/not_found)
- Análisis **thinking vs regular** automático

## 📁 Estructura Generada

```
plots/
├── comparisons/          # 🔥 PLOTS PRINCIPALES PARA PRESENTACIÓN
│   ├── 01_main_metrics_comparison.png      # Barras agrupadas de métricas clave
│   ├── 02_thinking_vs_regular_analysis.png # Análisis thinking vs regular
│   ├── 03_quality_speed_tradeoff.png       # Trade-off calidad vs velocidad
│   ├── 04_quality_distribution.png         # Distribución de niveles de calidad
│   ├── 05_performance_heatmap.png          # Heatmap de rendimiento
│   └── *_DOC.txt                          # Documentación de cada plot
├── agent_base/           # Análisis individual del agente base
├── agent_intention/      # Análisis individual del agente intention
├── agent_validation/     # Análisis individual del agente validation
├── agent_ensemble/       # Análisis individual del agente ensemble
├── EXECUTIVE_SUMMARY.txt # 📋 REPORTE EJECUTIVO CON INSIGHTS
└── run_analysis_examples.sh # 🔧 Script interactivo de ejemplos
```

## 📈 Plots Generados

### 🎯 Para Presentaciones (usar estos):
1. **01_main_metrics_comparison**: Comparación visual principal entre agentes
2. **02_thinking_vs_regular_analysis**: Impacto de modelos de reasoning
3. **03_quality_speed_tradeoff**: Relación calidad vs velocidad

### 🔍 Para Análisis Técnico:
4. **04_quality_distribution**: Consistencia de resultados por agente
5. **05_performance_heatmap**: Vista matricial de todas las métricas

### 📊 Análisis Individual:
- Distribución de tiempos de ejecución por agente
- Distribución de quality scores
- Estadísticas detalladas de rendimiento

## 🎯 Insights Clave Esperados

Basándose en los datos analizados, el sistema identificará:

- **Mejor agente overall**: Por success rate combinado
- **Trade-off thinking vs regular**: Velocidad vs calidad
- **Agente más eficiente**: Mejor balance calidad/tiempo
- **Patrones de consistencia**: Agentes especialistas vs generalistas

## 📋 Documentación

Cada plot incluye:
- **Archivo PNG**: Gráfico en alta resolución (300 DPI)
- **Archivo _DOC.txt**: Explicación detallada del plot
  - Qué muestra exactamente
  - Cómo interpretar los resultados
  - Cuándo usar en presentaciones

## 🔧 Personalización

El script es configurable para:
- Diferentes archivos JSON de resultados
- Detección automática de thinking models
- Estilos de visualización
- Filtros por agente o modelo

## 💡 Tips para Presentación

### Para Conferencias Académicas:
- Usar plots 01, 02, 03 como slides principales
- Incluir heatmap (05) para discusión técnica
- Referencias al EXECUTIVE_SUMMARY.txt para cifras exactas

### Para Proyecto Fin de Grado:
- Todos los plots son relevantes
- Usar análisis individuales para profundizar
- Documentación _DOC.txt como fuente para metodología

### Para Demos:
- Plot 02 (thinking vs regular) como hook principal
- Plot 03 (trade-off) para mostrar optimización
- Estadísticas del summary para conclusiones

## 🚨 Resolución de Problemas

**Error "archivo no encontrado":**
- Verificar que el archivo esté en `evaluation/results/`
- Usar nombre completo del archivo
- Probar con auto-detección: `python generate_analysis_plots.py`

**Estadísticas inesperadas:**
- El script calcula TODO desde cero desde el JSON raw
- Revisa el archivo JSON que estás especificando
- Compara con `EXECUTIVE_SUMMARY.txt` para ver qué se calculó

**Solo un agente en análisis:**
- Normal si el archivo JSON solo contiene un agente
- Para comparaciones, usa archivos con múltiples agentes

## 📊 Ejemplos de Uso

```bash
# Análisis rápido de 10 samples
python generate_analysis_plots.py results_local_10samples_phi4_14b_phi4_reasoning_14b_20250602_150120.json

# Análisis completo de 1500 samples  
python generate_analysis_plots.py results_local_1500samples_phi4_14b_phi4_reasoning_14b_20250603_081958.json

# Script interactivo
./run_analysis_examples.sh
``` 
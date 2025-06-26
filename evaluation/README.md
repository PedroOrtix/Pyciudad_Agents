# Sistema de Evaluación de Agentes PyCiudad

Un framework comprehensivo y organizado para evaluar, analizar y optimizar el rendimiento de agentes LLM en tareas de geolocalización.

## 🏗️ Estructura del Sistema

```
evaluation/
├── 📁 core/                    # Sistema principal de evaluación
│   ├── agent_evaluation_system.py
│   └── __init__.py
├── 📁 analysis/                # Análisis y comparación de métricas  
│   ├── metrics_analyzer.py
│   ├── compare_model_configs.py
│   └── __init__.py
├── 📁 configs/                 # Configuración de modelos y precios
│   ├── config_loader.py
│   ├── model_pricing.json
│   ├── update_model_config.py
│   └── __init__.py
├── 📁 scripts/                 # Scripts de ejecución
│   ├── run_evaluation.py
│   ├── example_evaluation.py
│   └── __init__.py
├── 📁 utils/                   # Utilidades y herramientas
│   ├── show_thinking_capabilities.py
│   └── __init__.py
├── 📁 results/                 # Resultados de evaluaciones
├── 📁 reports/                 # Reportes generados
└── requirements.txt
```

## 🚀 Scripts Principales (Raíz del Proyecto)

### Evaluación
```bash
# Evaluar agentes con configuración actual
python run_evaluation.py --samples 50 --concurrent 2

# Evaluar todo el dataset
python run_evaluation.py --full-dataset

# Solo analizar resultados existentes
python run_evaluation.py --analyze-only
```

### Análisis
```bash
# Analizar resultados más recientes
python analyze_results.py

# Comparación comprehensiva de configuraciones
python compare_model_configs.py --comprehensive

# Comparar configuraciones específicas
python compare_model_configs.py --config-names "baseline" "optimized"
```

## 📊 Módulos del Sistema

### 🔧 Core (`evaluation.core`)
Sistema principal de evaluación que conecta con LangGraph y ejecuta agentes contra datasets.

**Clase principal**: `AgentEvaluationSystem`

```python
from evaluation import AgentEvaluationSystem

# Crear evaluador
evaluator = AgentEvaluationSystem(
    langgraph_url="http://127.0.0.1:2024",
    dataset_path="data/datasets/dataset_generado_final.json"
)

# Ejecutar evaluación
results = await evaluator.evaluate_all_agents(max_samples=100)
```

### 📈 Analysis (`evaluation.analysis`)
Herramientas avanzadas para análisis de métricas y comparación de configuraciones.

**Clases principales**: 
- `MetricsAnalyzer`: Análisis detallado de una evaluación
- `ComprehensiveModelComparator`: Comparación entre múltiples configuraciones

```python
from evaluation import MetricsAnalyzer, ComprehensiveModelComparator

# Análisis de métricas
analyzer = MetricsAnalyzer("evaluation/results/evaluation_results_20241220_143022.json")
analyzer.load_results()
report = analyzer.generate_performance_report()

# Comparación de configuraciones
comparator = ComprehensiveModelComparator()
comparator.discover_all_evaluations()
comprehensive_report = comparator.generate_comprehensive_report()
```

### ⚙️ Configs (`evaluation.configs`)
Gestión de configuraciones de modelos, precios y parámetros.

```python
from evaluation import get_agent_model_config, get_model_pricing

# Obtener configuración de agentes
config = get_agent_model_config()

# Obtener precios de modelos
pricing = get_model_pricing()
```

### 🔧 Utils (`evaluation.utils`)
Utilidades auxiliares y herramientas de desarrollo.

## 📋 Flujo de Trabajo Típico

### 1. Configurar Modelos
```bash
# Configurar modelos de Ollama
export OPENAI_MODEL_NAME="qwen3:30b-a3b"
export ANTHROPIC_MODEL_NAME="qwq:latest"
export OPENAI_BASE_URL="http://localhost:11434/v1"
export ANTHROPIC_BASE_URL="http://localhost:11434/v1"
export OPENAI_API_KEY="ollama"
export ANTHROPIC_API_KEY="ollama"
```

### 2. Ejecutar Evaluación
```bash
python run_evaluation.py --samples 50 --concurrent 2
```

### 3. Analizar Resultados
```bash
python analyze_results.py
```

### 4. Comparar Configuraciones
```bash
# Cambiar modelos y repetir evaluación
export OPENAI_MODEL_NAME="phi4:14b-q8_0"
export ANTHROPIC_MODEL_NAME="phi4-reasoning:14b-plus-q8_0"
python run_evaluation.py --samples 50

# Comparar todas las configuraciones
python compare_model_configs.py --comprehensive
```

## 📈 Outputs del Sistema

### Evaluación Individual
- `evaluation/results/evaluation_results_TIMESTAMP.json`: Resultados completos
- `evaluation/performance_report.json`: Reporte de rendimiento
- `evaluation/detailed_results.csv`: Datos detallados en CSV
- `evaluation/plots/`: Visualizaciones

### Comparación de Configuraciones
- `evaluation/comprehensive_analysis/comprehensive_analysis_report_TIMESTAMP.json`: Reporte completo
- `evaluation/comprehensive_analysis/complete_analysis_data_TIMESTAMP.csv`: Datos completos
- `evaluation/comprehensive_analysis/plots/`: Visualizaciones avanzadas
  - `metrics_heatmap.png`: Matriz de comparación
  - `radar_comparison.png`: Gráfico radar
  - `rankings_by_metric.png`: Rankings
  - `efficiency_analysis.png`: Análisis de eficiencia
  - `temporal_evolution.png`: Evolución temporal

## 🔧 Configuración

### Modelos Soportados
El sistema está optimizado para **Ollama** (modelos locales), pero también soporta APIs externas:

**Modelos Ollama disponibles en tu sistema:**
- **Qwen**: `qwen3:30b-a3b` (modelo MOE eficiente)
- **QwQ (Reasoning)**: `qwq:latest` (modelo de razonamiento de 32B)
- **Phi4**: `phi4:14b-q8_0` (modelo base de Microsoft)
- **Phi4 Reasoning**: `phi4-reasoning:14b-plus-q8_0` (modelo de razonamiento)
- **Gemma3**: `gemma3:27b-it-qat` (modelo de razonamiento de Google)

**APIs externas (opcional):**
- **OpenAI**: GPT-4, GPT-3.5, etc.
- **Anthropic**: Claude series
- **Google**: Gemini series  
- **Groq**: Llama, Mixtral, etc.

### Variables de Entorno
```bash
# Configuración para Ollama (modelos locales)
# Modelo regular para todos los agentes
OLLAMA_MODEL="qwen3:30b-a3b"
OLLAMA_MODEL_THINKING="qwq:latest"
```

### Agentes Evaluados
- `agent_base`: Agente básico de geolocalización
- `agent_intention`: Con análisis de intención
- `agent_validation`: Con validación de resultados
- `agent_ensemble`: Conjunto de agentes

## 📊 Métricas Principales

### Rendimiento
- **Tasa de éxito**: Porcentaje de ejecuciones exitosas
- **Tiempo de ejecución**: Estadísticas de tiempo por query
- **Throughput**: Muestras procesadas por segundo

### Costos
- **Tokens utilizados**: Input, output y total
- **Costos estimados**: Por query y total
- **Eficiencia**: Ratio éxito/costo

### Calidad
- **Precisión geográfica**: Exactitud de geolocalización
- **Consistencia**: Variabilidad entre ejecuciones
- **Robustez**: Manejo de casos edge

## 🚀 Extensibilidad

El sistema está diseñado para ser extensible:

### Agregar Nuevos Agentes
1. Crear el agente en `agents/`
2. Agregarlo a `langgraph.json`
3. Incluirlo en `AgentEvaluationSystem.agents`

### Agregar Nuevas Métricas
1. Extender `MetricsAnalyzer._extract_token_metrics()`
2. Actualizar visualizaciones en `create_visualizations()`

### Agregar Nuevos Modelos
1. Actualizar `evaluation/configs/model_pricing.json`
2. Configurar variables de entorno correspondientes

## 🛠️ Desarrollo

### Requisitos Previos
```bash
# 1. Ollama debe estar ejecutándose
ollama serve

# 2. Descargar modelos necesarios
ollama pull qwen3:30b-a3b
ollama pull qwq:latest
ollama pull phi4:14b-q8_0
ollama pull phi4-reasoning:14b-plus-q8_0
ollama pull gemma3:27b-it-qat

# 3. Verificar modelos disponibles
ollama list
```

### Instalación
```bash
pip install -r requirements.txt
```

### Verificar Configuración de Ollama
```bash
# Verificar que Ollama esté funcionando
curl http://localhost:11434/api/tags

# Probar un modelo específico
curl http://localhost:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3:30b-a3b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Testing
```bash
# Evaluación rápida de prueba
python run_evaluation.py --samples 5 --concurrent 1

# Verificar configuración
python -c "from evaluation import get_agent_model_config; print(get_agent_model_config())"
```

### Debugging
```bash
# Mostrar capacidades de thinking
python evaluation/utils/show_thinking_capabilities.py

# Verificar configuración de modelos
python evaluation/configs/update_model_config.py

# Verificar conectividad con Ollama
python -c "
import requests
try:
    r = requests.get('http://localhost:11434/api/tags')
    print('✅ Ollama conectado:', r.status_code)
    print('Modelos disponibles:', [m['name'] for m in r.json()['models']])
except:
    print('❌ Error conectando con Ollama')
"
```


## 📝 Licencia

Ver `LICENSE` en la raíz del proyecto.
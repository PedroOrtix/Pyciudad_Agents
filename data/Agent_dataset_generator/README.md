# Agente Generador de Datasets

Este agente está diseñado para generar datasets de entrenamiento basados en el archivo de ground truth `dataset_direcciones_init.json`.

## Funcionalidad

El agente toma las direcciones del ground truth y genera múltiples variaciones de consultas de usuario que un usuario real podría hacer para buscar cada dirección, incluyendo:

- **Consultas en lenguaje natural**: Con diferentes niveles de formalidad
- **Errores ortográficos**: Simulando errores típicos de usuarios reales
- **Diferentes tipos de consulta**: Directas, naturales, coloquiales y preguntas
- **Niveles de dificultad**: Fácil, medio y alto

## Estructura

### Archivos principales:

- `agent_dataset_generator.py`: Implementación principal del agente
- `states_dataset.py`: Definición de estados del grafo
- `prompt_dataset.py`: Prompts para la generación de variaciones
- `langgraph.json`: Configuración de LangGraph

### Flujo del agente:

1. **Load Ground Truth**: Carga las direcciones del archivo JSON de referencia
2. **Generate Natural Queries**: Genera 5 variaciones por cada dirección
3. **Inject Errors**: Inyecta errores ortográficos en consultas de dificultad media/alta
4. **Save Dataset**: Guarda el dataset final con estadísticas

## Uso

### Desde línea de comandos:

```bash
# Procesar 10 direcciones (por defecto)
python data/run_dataset_generator.py

# Procesar un número específico
python data/run_dataset_generator.py --sample-size 50

# Procesar todas las direcciones
python data/run_dataset_generator.py --all

# Especificar archivo de salida
python data/run_dataset_generator.py --output mi_dataset.json
```

### Desde código:

```python
from data.Agent_dataset_generator.agent_dataset_generator import app_dataset_generator
from data.Agent_dataset_generator.states_dataset import GraphStateInput

# Configurar parámetros
initial_state = GraphStateInput(
    sample_size=100,
    output_filename="dataset_personalizado.json",
    variations_per_address=5
)

# Ejecutar agente
result = await app_dataset_generator.ainvoke(initial_state)
```

## Salida

El agente genera un archivo JSON con la siguiente estructura para cada entrada:

```json
{
  "ground_truth_id": "150010000013",
  "ground_truth_address": "LUGAR CALLE (Abegondo)",
  "user_query": "cómo llego a lugar calle en abegondo",
  "query_type": "pregunta",
  "difficulty_level": "medio",
  "has_errors": false,
  "error_types": [],
  "original_clean_query": "cómo llego a lugar calle en abegondo",
  "ground_truth_data": {
    // Datos completos del ground truth original
  }
}
```

## Estadísticas

El agente proporciona estadísticas detalladas:

- Total de entradas generadas
- Número de consultas con/sin errores
- Distribución por nivel de dificultad
- Distribución por tipo de consulta

## Tipos de Variaciones

### Tipos de consulta:
- **Directa**: La dirección tal como aparece
- **Natural**: Lenguaje más coloquial
- **Coloquial**: Expresiones cotidianas
- **Pregunta**: Formulada como pregunta explícita

### Niveles de dificultad:
- **Fácil**: Consulta clara y directa
- **Medio**: Con ambigüedad o términos coloquiales
- **Alto**: Muy coloquial, con posibles ambigüedades

### Tipos de errores inyectados:
- Errores ortográficos (tildes, letras intercambiadas)
- Errores de mayúsculas/minúsculas
- Abreviaciones incorrectas
- Espaciado incorrecto
- Sustituciones de caracteres especiales 
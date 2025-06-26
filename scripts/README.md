# 🛠️ Scripts Utilitarios

Este directorio contiene scripts auxiliares y utilitarios del proyecto PyCiudad_Agents, organizados por funcionalidad.

## 📁 Estructura

```
scripts/
├── cleanup_redundant_files.py     # Script de limpieza de archivos redundantes
├── verify_dependencies.py         # Verificación de dependencias
├── data/                          # Scripts relacionados con datasets
│   ├── generar_dataset_direcciones.py
│   └── run_dataset_generator.py
└── README.md                      # Este archivo
```

## 🗑️ cleanup_redundant_files.py

Script para limpiar archivos redundantes del sistema de evaluación.

**Uso:**
```bash
# Desde la raíz del proyecto
python scripts/cleanup_redundant_files.py --dry-run    # Ver qué se eliminaría
python scripts/cleanup_redundant_files.py             # Eliminar realmente
```

**Funcionalidad:**
- Elimina reports redundantes
- Limpia plots antiguos (se regeneran automáticamente)
- Elimina cache de Python (`__pycache__`)
- Limpia checkpoints antiguos (>7 días)

## ✅ verify_dependencies.py

Script para verificar que todas las dependencias consolidadas están instaladas correctamente.

**Uso:**
```bash
# Desde la raíz del proyecto
python scripts/verify_dependencies.py
```

**Funcionalidad:**
- Verifica todas las dependencias principales del proyecto
- Categoriza dependencias por funcionalidad (CORE, EVALUATION, ANALYSIS, etc.)
- Proporciona información detallada sobre dependencias faltantes
- Código de salida 0 si todo está OK, 1 si faltan dependencias

## 📊 data/generar_dataset_direcciones.py

Script para generar el dataset inicial de direcciones desde CartoCiudad.

**Uso:**
```bash
# Desde la raíz del proyecto
python scripts/data/generar_dataset_direcciones.py
```

**Funcionalidad:**
- Busca direcciones por todas las provincias de España
- Usa términos como "calle", "avenida", "plaza", "paseo"
- Deduplica resultados
- Genera `data/datasets/dataset_direcciones_init.json`

## 🔄 data/run_dataset_generator.py

Script para ejecutar el agente generador de datasets de entrenamiento.

**Uso:**
```bash
# Desde la raíz del proyecto
python scripts/data/run_dataset_generator.py --sample-size 50
python scripts/data/run_dataset_generator.py --all                    # Todas las direcciones
python scripts/data/run_dataset_generator.py --output mi_dataset.json
```

**Funcionalidad:**
- Ejecuta el Agent_dataset_generator
- Genera variaciones de consultas de usuario
- Inyecta errores simulados
- Crea datasets de entrenamiento con estadísticas

## 🔗 Uso General

Todos los scripts están diseñados para ejecutarse desde la **raíz del proyecto** para mantener las rutas de importación correctas.

**Ejemplo:**
```bash
cd /ruta/a/PyCiudad_Agents
python scripts/cleanup_redundant_files.py
python scripts/data/generar_dataset_direcciones.py
```

## 📋 Dependencias

Los scripts utilizan las mismas dependencias del proyecto principal definidas en `requirements.txt`. 
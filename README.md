# 🚀 PyCiudad Agents

Arquitectura modular para el desarrollo de agentes inteligentes en Python, diseñada para facilitar la extensión y reutilización de componentes en proyectos de IA y automatización.

---

## 📦 Estructura del Proyecto

- `agents/` — Código fuente de los agentes y utilidades comunes:
  - `Agent_base/` — Implementación base de agentes.
  - `Agent_ensemble/` — Agentes de tipo ensemble.
  - `Agent_intention/` — Agentes basados en intención.
  - `Agent_validation/` — Agentes de validación.
  - `common/` — Utilidades, esquemas y herramientas compartidas.
- `examples/` — Ejemplos de uso de los agentes.
- `requirements.txt` — Dependencias del proyecto.

---

## ⚙️ Instalación

### 1. Clona el repositorio

```bash
git clone https://github.com/tu_usuario/PyCiudad_Agents.git
cd PyCiudad_Agents
```

### 2. Crea y activa un entorno conda (Python 3.11)

```bash
conda create -n pyciudad-agents python=3.11 -y
conda activate pyciudad-agents
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

---

## 🧑‍💻 Uso

Consulta los scripts en la carpeta `examples/` para ver ejemplos de cómo invocar y utilizar los agentes. Por ejemplo:

```bash
python examples/run_agente_base.py
python examples/run_agente_intention.py
python examples/run_agente_validation.py
python examples/run_agente_ensemble.py
```

---

## 🟢 Servicio LangSmith (LangGraph)

Para levantar el servicio de desarrollo de LangSmith, ejecuta:

```bash
langgraph dev
```

Asegúrate de tener instalado `langgraph` y configurado tu entorno según la documentación oficial.

---

## 📄 Licencia

Este proyecto está bajo la licencia MIT.

---

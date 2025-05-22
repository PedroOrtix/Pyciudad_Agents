# PyCiudad Agents

Este repositorio contiene una arquitectura modular para el desarrollo de agentes inteligentes en Python, diseñada para facilitar la extensión y reutilización de componentes en proyectos de inteligencia artificial y automatización.

## Estructura del Proyecto

- `agents/`: Código fuente de los agentes y utilidades comunes.
  - `Agent_base/`: Implementación base de agentes.
  - `Agent_ensemble/`: Agentes de tipo ensemble.
  - `Agent_intention/`: Agentes basados en intención.
  - `Agent_validation/`: Agentes de validación.
  - `common/`: Utilidades, esquemas y herramientas compartidas.
- `deployment/`: Archivos para despliegue (Docker, API, etc).
- `examples/`: Ejemplos de uso de los agentes.
- `requirements.txt`: Dependencias del proyecto.

## Instalación

Clona el repositorio y luego instala las dependencias:

```bash
git clone https://github.com/tu_usuario/PyCiudad_Agents.git
cd PyCiudad_Agents
pip install -r requirements.txt
```

## Uso

Consulta los scripts en la carpeta `examples/` para ver ejemplos de cómo invocar y utilizar los agentes. Por ejemplo:

```bash
python examples/run_agente_base.py
```

## Licencia

Este proyecto está bajo la licencia MIT.

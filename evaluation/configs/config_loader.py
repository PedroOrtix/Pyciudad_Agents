"""
Cargador de Configuración Dinámica para Evaluación de Agentes

Este módulo lee las configuraciones de modelos y precios desde variables
de entorno y archivos de configuración, permitiendo flexibilidad total
para modelos locales y cambiantes.
"""

import os
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

def load_model_config() -> Dict[str, Any]:
    """
    Cargar configuración de modelos desde variables de entorno y archivos.
    
    Returns:
        Dict con configuración de agentes y precios
    """
    
    # Obtener modelos desde .env
    regular_model = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b")
    thinking_model = os.getenv("OLLAMA_MODEL_THINKING", "qwq:latest")
    
    print(f"🔧 Modelos detectados desde .env:")
    print(f"   📝 Modelo regular: {regular_model}")
    print(f"   🧠 Modelo thinking: {thinking_model}")
    
    # Configuración de agentes (dinámica basada en modelos actuales)
    agent_model_config = {
        "agent_base": {
            "has_thinking_model": False,
            "models": ["llm"],
            "actual_models": {
                "llm": regular_model
            }
        },
        "agent_intention": {
            "has_thinking_model": True,
            "models": ["llm_thinking", "llm"],
            "actual_models": {
                "llm_thinking": thinking_model,
                "llm": regular_model
            }
        },
        "agent_validation": {
            "has_thinking_model": True,
            "models": ["llm_thinking", "llm"],
            "actual_models": {
                "llm_thinking": thinking_model,
                "llm": regular_model
            }
        },
        "agent_ensemble": {
            "has_thinking_model": True,
            "models": ["llm_thinking", "llm"],
            "actual_models": {
                "llm_thinking": thinking_model,
                "llm": regular_model
            }
        }
    }
    
    return {
        "agent_model_config": agent_model_config,
        "current_models": {
            "regular": regular_model,
            "thinking": thinking_model
        }
    }

def load_model_pricing() -> Dict[str, Dict[str, float]]:
    """
    Cargar precios de modelos desde archivo de configuración o defaults.
    
    Returns:
        Dict con precios por modelo (USD por 1M tokens)
    """
    
    # Intentar cargar desde archivo personalizado
    pricing_file = Path("evaluation/model_pricing.json")
    
    if pricing_file.exists():
        print(f"📊 Cargando precios desde: {pricing_file}")
        with open(pricing_file, 'r') as f:
            return json.load(f)
    
    # Precios por defecto (USD por 1M tokens - estándar de la industria)
    # Fuente: DeepInfra (plataforma unificada para evitar sesgos de precios)
    default_pricing = {
    "qwen3:30b-a3b": {
        "input": 0.08,
        "output": 0.29,
        "description": "Modelo MOE de Qwen con 30B de parametros pero 3B de parametros activos (DeepInfra)"
    },
    "qwq:latest": {
        "input": 0.15,
        "output": 0.2,
        "description": "Modelo Razonador de Qwen con 32B de parametros (DeepInfra)"
    },
    "phi4:14b-q8_0": {
        "input": 0.07,
        "output": 0.14,
        "description": "Modelo Phi4 de Microsoft con 14B de parametros (DeepInfra)"
    },
    "phi4-reasoning:14b-plus-q8_0": {
        "input": 0.07,
        "output": 0.35,
        "description": "Modelo Razonador de Phi4 de Microsoft con 14B de parametros (DeepInfra)"
    },
    "gemma3:27b-it-qat": {
        "input": 0.10,
        "output": 0.20,
        "description": "Modelo Razonador de Gemma3 de Google con 27B de parametros (DeepInfra)"
    }
    }
    
    print(f"📊 Usando precios por defecto desde DeepInfra (puedes personalizarlos en {pricing_file})")
    return default_pricing

def get_model_price(model_name: str, pricing: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Obtener precio para un modelo específico con fallbacks inteligentes.
    
    Args:
        model_name: Nombre del modelo
        pricing: Dict de precios
        
    Returns:
        Dict con precios input/output
    """
    
    # Busqueda exacta
    if model_name in pricing:
        return pricing[model_name]
    
    # Busqueda por patrones (para modelos locales con versiones)
    model_lower = model_name.lower()
    
    for pattern, price in pricing.items():
        if pattern in model_lower:
            print(f"💡 Precio para '{model_name}' basado en patrón '{pattern}': {price}")
            return price
    
    # Fallback a default
    print(f"⚠️  Modelo '{model_name}' no encontrado, usando precio default")
    return pricing["default"]

def save_model_pricing_template():
    """Crear archivo template para personalizar precios"""
    
    pricing_file = Path("evaluation/model_pricing.json")
    
    if pricing_file.exists():
        print(f"📄 Template de precios ya existe: {pricing_file}")
        return str(pricing_file)
    
    # Crear directorio si no existe
    pricing_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Obtener modelos actuales del .env
    config = load_model_config()
    current_models = config["current_models"]
    
    # Template basado en modelos actuales
    template = {
        "# Comentarios": "Precios en USD por 1M tokens desde DeepInfra (plataforma unificada para evitar sesgos)",
        "# Fuente": "https://deepinfra.com/pricing - Ajusta según tus costos reales",
        
        # Modelos actuales detectados
        current_models["regular"]: {
            "input": 0.08,
            "output": 0.29,
            "description": "Tu modelo regular actual (precio base DeepInfra)"
        },
        current_models["thinking"]: {
            "input": 0.15,
            "output": 0.2,
            "description": "Tu modelo thinking actual (precio base DeepInfra)"
        },
        
        # Algunos ejemplos comunes (precios DeepInfra)
        "default": {"input": 0.1, "output": 0.2},
        "qwen": {"input": 0.08, "output": 0.29},
        "llama": {"input": 0.07, "output": 0.28},
        "mistral": {"input": 0.06, "output": 0.24}
    }
    
    with open(pricing_file, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"📄 Template de precios creado: {pricing_file}")
    print(f"💡 Personaliza los precios según tus modelos y costos reales")
    
    return str(pricing_file)

def update_evaluation_config(results_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Actualizar metadata de resultados con configuración actual.
    
    Args:
        results_metadata: Metadata existente
        
    Returns:
        Metadata actualizada con configuración de modelos
    """
    
    config = load_model_config()
    pricing = load_model_pricing()
    
    # Agregar configuración actual a los metadatos
    results_metadata.update({
        "model_configuration": {
            "agent_model_config": config["agent_model_config"],
            "current_models": config["current_models"],
            "pricing_source": "custom" if Path("evaluation/model_pricing.json").exists() else "default",
            "timestamp": results_metadata.get("evaluation_timestamp")
        }
    })
    
    return results_metadata

# Funciones de compatibilidad para el código existente
def get_agent_model_config() -> Dict[str, Any]:
    """Compatibilidad: obtener configuración de agentes"""
    return load_model_config()["agent_model_config"]

def get_model_pricing() -> Dict[str, Dict[str, float]]:
    """Compatibilidad: obtener precios de modelos"""
    return load_model_pricing()

if __name__ == "__main__":
    print("🔧 Configuración de Evaluación - Test")
    print("=" * 50)
    
    config = load_model_config()
    pricing = load_model_pricing()
    
    print(f"\n🤖 Agentes configurados: {len(config['agent_model_config'])}")
    for agent, conf in config['agent_model_config'].items():
        print(f"   {agent}: thinking={conf['has_thinking_model']}")
    
    print(f"\n💰 Modelos con precio: {len(pricing)}")
    
    # Crear template si no existe
    template_path = save_model_pricing_template()
    print(f"\n📄 Template disponible en: {template_path}") 
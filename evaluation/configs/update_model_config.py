#!/usr/bin/env python3
"""
Script para Gestionar Configuración de Modelos

Este script te ayuda a gestionar los cambios de modelos locales
y actualizar las configuraciones de evaluación correspondientes.
"""

import os
import json
import sys # Asegurar que imports relativos funcionen
from pathlib import Path
from dotenv import load_dotenv, set_key

# Añadir el directorio padre al path para imports relativos
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Ajustar paths para que funcionen desde evaluation/
ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
PRICING_FILE_PATH = Path(__file__).resolve().parent / "model_pricing.json"

def show_current_config():
    """Mostrar configuración actual de modelos"""
    load_dotenv(ENV_FILE)
    
    regular_model = os.getenv("OLLAMA_MODEL", "qwen3:30b-a3b")
    thinking_model = os.getenv("OLLAMA_MODEL_THINKING", "qwq:latest")
    
    print("🔧 CONFIGURACIÓN ACTUAL DE MODELOS")
    print("=" * 50)
    print(f"📝 Modelo Regular (OLLAMA_MODEL): {regular_model}")
    print(f"🧠 Modelo Thinking (OLLAMA_MODEL_THINKING): {thinking_model}")
    
    if PRICING_FILE_PATH.exists():
        print(f"💰 Archivo de precios: {PRICING_FILE_PATH}")
        with open(PRICING_FILE_PATH, 'r') as f:
            pricing = json.load(f)
            
        print("📊 Modelos con precio configurado:")
        for model, price in pricing.items():
            if isinstance(price, dict) and "input" in price:
                print(f"   • {model}: ${price['input']:.6f}/${price['output']:.6f}")
    else:
        print("💰 Archivo de precios: No configurado (usando defaults en config_loader.py)")
        print(f"   (Se creará en {PRICING_FILE_PATH} si actualizas precios)")

def update_env_models(regular_model: str = None, thinking_model: str = None):
    """Actualizar modelos en el .env"""
    
    if not ENV_FILE.exists():
        print(f"⚠️  Archivo .env no encontrado en {ENV_FILE}, creando uno nuevo...")
        ENV_FILE.touch()
    
    if regular_model:
        set_key(str(ENV_FILE), "OLLAMA_MODEL", regular_model)
        print(f"✅ Actualizado OLLAMA_MODEL en {ENV_FILE}: {regular_model}")
    
    if thinking_model:
        set_key(str(ENV_FILE), "OLLAMA_MODEL_THINKING", thinking_model)
        print(f"✅ Actualizado OLLAMA_MODEL_THINKING en {ENV_FILE}: {thinking_model}")

def update_model_pricing():
    """Actualizar precios de modelos personalizados"""
    
    print("💰 ACTUALIZAR PRECIOS DE MODELOS")
    print("=" * 40)
    print("📊 Fuente recomendada: DeepInfra (plataforma unificada)")
    print("🔗 https://deepinfra.com/pricing")
    print()
    
    pricing_file = Path("evaluation/model_pricing.json")
    
    # Cargar precios existentes o crear nuevos
    if pricing_file.exists():
        with open(pricing_file, 'r') as f:
            pricing = json.load(f)
        print(f"📄 Precios existentes cargados desde: {pricing_file}")
    else:
        pricing = {}
        print(f"📄 Creando nuevo archivo de precios: {pricing_file}")
    
    print("\n💡 Formato: USD por 1M tokens (estándar DeepInfra)")
    print("💡 Deja vacío para mantener valor actual")
    print()
    
    # Obtener modelos actuales
    config = load_model_config()
    current_models = config["current_models"]
    
    models_to_update = [
        current_models["regular"],
        current_models["thinking"]
    ]
    
    for model in models_to_update:
        print(f"\n🔧 Configurando precios para: {model}")
        
        current_price = pricing.get(model, {"input": 0.0, "output": 0.0})
        
        print(f"   Precio actual input: ${current_price.get('input', 0.0):.6f}")
        print(f"   Precio actual output: ${current_price.get('output', 0.0):.6f}")
        
        try:
            input_price = input("   Nuevo precio input ($/1M tokens): ").strip()
            output_price = input("   Nuevo precio output ($/1M tokens): ").strip()
            description = input("   Descripción (opcional): ").strip()
            
            if input_price or output_price or description:
                if model not in pricing:
                    pricing[model] = {}
                
                if input_price:
                    pricing[model]["input"] = float(input_price)
                if output_price:
                    pricing[model]["output"] = float(output_price)
                if description:
                    pricing[model]["description"] = f"{description} (DeepInfra)"
                elif "description" not in pricing[model]:
                    pricing[model]["description"] = "Modelo personalizado (DeepInfra)"
                
                print(f"   ✅ Precios actualizados para {model}")
            else:
                print(f"   ⏭️  Sin cambios para {model}")
                
        except ValueError:
            print(f"   ❌ Error: Precio inválido para {model}")
            continue
        except KeyboardInterrupt:
            print("\n⏹️  Actualización cancelada")
            return
    
    # Agregar metadata
    pricing["# Comentarios"] = "Precios en USD por 1M tokens desde DeepInfra (plataforma unificada para evitar sesgos)"
    pricing["# Fuente"] = "https://deepinfra.com/pricing - Precios actualizados para comparación justa"
    
    # Guardar archivo
    pricing_file.parent.mkdir(parents=True, exist_ok=True)
    with open(pricing_file, 'w') as f:
        json.dump(pricing, f, indent=2)
    
    print(f"\n💾 Precios guardados en: {pricing_file}")
    print("📊 Fuente: DeepInfra (comparación justa sin sesgos)")
    
    return str(pricing_file)

def test_configuration():
    """Probar la configuración actual (usa config_loader)"""
    
    print("\n🧪 PROBANDO CONFIGURACIÓN (usando config_loader.py)...")
    print("=" * 55)
    
    try:
        # Importar aquí para asegurar que sys.path está actualizado
        from evaluation.configs.config_loader import load_model_config, load_model_pricing, get_model_price
        
        config = load_model_config() # Esto imprimirá los modelos detectados desde .env
        pricing = load_model_pricing() # Esto imprimirá la fuente de precios
        
        print("\n✅ Configuración cargada exitosamente por config_loader")
        print(f"📊 Agentes configurados: {len(config['agent_model_config'])}")
        print(f"💰 Modelos con precio disponibles: {len(pricing)}")
        
        current_models = config["current_models"]
        print("\n🔍 Modelos actualmente en uso (según config_loader):")
        print(f"   Regular: {current_models['regular']}")
        print(f"   Thinking: {current_models['thinking']}")
        
        regular_price = get_model_price(current_models['regular'], pricing)
        thinking_price = get_model_price(current_models['thinking'], pricing)
        
        print("\n💰 Precios asignados por config_loader:")
        print(f"   {current_models['regular']}: ${regular_price['input']:.6f}/${regular_price['output']:.6f}")
        print(f"   {current_models['thinking']}: ${thinking_price['input']:.6f}/${thinking_price['output']:.6f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en configuración vía config_loader: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_setup():
    """Setup interactivo para configurar modelos"""
    
    print("🛠️  CONFIGURACIÓN INTERACTIVA DE MODELOS (Script: evaluation/update_model_config.py)")
    print("=" * 70)
    
    show_current_config()
    
    print("\n¿Qué quieres hacer?")
    print("1. Cambiar modelo regular (OLLAMA_MODEL en .env)")
    print("2. Cambiar modelo thinking (OLLAMA_MODEL_THINKING en .env)")
    print("3. Actualizar/Añadir precios en evaluation/model_pricing.json")
    print("4. Probar configuración actual (vía config_loader.py)")
    print("5. Mostrar ayuda")
    print("0. Salir")
    
    choice = input("\nOpción: ").strip()
    
    if choice == "1":
        new_model = input(f"Nuevo modelo regular (actual: {os.getenv('OLLAMA_MODEL')}, dejar vacío para no cambiar): ").strip()
        if new_model:
            update_env_models(regular_model=new_model)
            if input(f"¿Configurar precio para '{new_model}' en {PRICING_FILE_PATH}? (y/n): ").lower() == 'y':
                try:
                    input_price = float(input("Precio input (USD por 1K tokens): "))
                    output_price = float(input("Precio output (USD por 1K tokens): "))
                    update_model_pricing(new_model, input_price, output_price)
                except ValueError:
                    print("⚠️  Precios inválidos. Actualiza manualmente el JSON si es necesario.")
    
    elif choice == "2":
        new_model = input(f"Nuevo modelo thinking (actual: {os.getenv('OLLAMA_MODEL_THINKING')}, dejar vacío para no cambiar): ").strip()
        if new_model:
            update_env_models(thinking_model=new_model)
            if input(f"¿Configurar precio para '{new_model}' en {PRICING_FILE_PATH}? (y/n): ").lower() == 'y':
                try:
                    input_price = float(input("Precio input (USD por 1K tokens): "))
                    output_price = float(input("Precio output (USD por 1K tokens): "))
                    update_model_pricing(new_model, input_price, output_price)
                except ValueError:
                    print("⚠️  Precios inválidos. Actualiza manualmente el JSON si es necesario.")
    
    elif choice == "3":
        model_name = input("Nombre del modelo a actualizar/añadir en JSON: ").strip()
        if model_name:
            try:
                input_price = float(input("Precio input (USD por 1K tokens): "))
                output_price = float(input("Precio output (USD por 1K tokens): "))
                update_model_pricing(model_name, input_price, output_price)
            except ValueError:
                print("❌ Precios inválidos.")
        else:
            print("⚠️  Nombre de modelo vacío.")

    elif choice == "4":
        test_configuration()
    
    elif choice == "5":
        show_help()
    
    elif choice == "0":
        print("👋 ¡Hasta luego!")
        return
    else:
        print("❌ Opción no válida.")

    if choice != "0":
        input("\nPresiona Enter para continuar...")
        interactive_setup()

def show_help():
    """Mostrar ayuda sobre el sistema"""
    
    print("\n📚 AYUDA - GESTIÓN DE MODELOS LOCALES (evaluation/update_model_config.py)")
    print("=" * 70)
    
    print("\nEste script gestiona:")
    print(f"  - Variables OLLAMA_MODEL y OLLAMA_MODEL_THINKING en {ENV_FILE}")
    print(f"  - Archivo de precios {PRICING_FILE_PATH}")
    
    print("\n🔄 Flujo típico para cambiar modelos:")
    print("1. Usa este script (o edita manualmente) para cambiar modelos en .env")
    print("2. Usa este script (o edita manualmente) para configurar precios en el JSON")
    print("3. Prueba con 'python evaluation/update_model_config.py test' o 'python evaluation/config_loader.py'")
    print("4. Ejecuta tus evaluaciones (ej: python example_evaluation.py, python run_evaluation.py)")
    
    print("\n📁 Archivos Clave:")
    print(f"   • {ENV_FILE.name} (en el directorio raíz del proyecto): Define modelos activos")
    print(f"   • {PRICING_FILE_PATH.name}: Define precios personalizados. Si no existe, se usan defaults.")
    print("   • evaluation/config_loader.py: Lógica central que lee .env y el JSON de precios.")
    
    print("\n🚀 Comandos útiles:")
    print("   python evaluation/update_model_config.py          # Interactivo (este script)")
    print("   python evaluation/update_model_config.py show     # Ver config actual")
    print("   python evaluation/update_model_config.py test     # Probar config con config_loader")
    print("   python evaluation/config_loader.py                # Test directo de config_loader")
    print("   python example_evaluation.py                      # Demo con modelos actuales")

def main():
    """Función principal"""
    
    # Asegurar que estamos en el contexto del proyecto para dotenv y paths relativos
    project_root = Path(__file__).resolve().parent.parent
    os.chdir(project_root) # Cambiar CWD al root del proyecto temporalmente
    # Esto asegura que load_dotenv() en config_loader funcione bien sin path explícito
    # y que los paths relativos en otros módulos sean consistentes.

    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "show":
            show_current_config()
        elif command == "test":
            test_configuration()
        elif command == "help":
            show_help()
        else:
            print(f"❌ Comando desconocido: {command}")
            print("Comandos disponibles: show, test, help, o ninguno para interactivo.")
    else:
        interactive_setup()

if __name__ == "__main__":
    main() 
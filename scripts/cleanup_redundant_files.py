#!/usr/bin/env python3
"""
Script para limpiar archivos redundantes del sistema de evaluación.

Este script elimina archivos innecesarios que se generan automáticamente
y que son redundantes una vez que el sistema de análisis puede trabajar
directamente con los JSON crudos de resultados.

Uso: python cleanup_redundant_files.py [--dry-run]
"""

import argparse
from pathlib import Path
from typing import List, Tuple

def find_redundant_files() -> List[Tuple[Path, str]]:
    """
    Encuentra todos los archivos redundantes en el proyecto.
    
    Returns:
        Lista de tuplas (archivo, descripción)
    """
    redundant_files = []
    
    # 1. Directorio completo de reports
    reports_dir = Path("evaluation/reports")
    if reports_dir.exists():
        for file_path in reports_dir.rglob("*"):
            if file_path.is_file():
                if file_path.name.startswith("performance_report_"):
                    redundant_files.append((file_path, "Performance Report (REDUNDANTE)"))
                elif file_path.name.startswith("detailed_results_"):
                    redundant_files.append((file_path, "Detailed Results CSV (REDUNDANTE)"))
                else:
                    redundant_files.append((file_path, "Archivo en directorio reports (INNECESARIO)"))
    
    # 2. Plots antiguos (se regeneran automáticamente)
    plots_dir = Path("plots")
    if plots_dir.exists():
        for subdir in ["agent_base", "agent_intention", "agent_validation", 
                      "agent_ensemble", "comparisons", "performance_based"]:
            subdir_path = plots_dir / subdir
            if subdir_path.exists():
                for file_path in subdir_path.rglob("*"):
                    if file_path.is_file():
                        redundant_files.append((file_path, f"Plot en {subdir} (SE REGENERA)"))
    
    # 3. Archivos de cache de pycache
    for pycache_dir in Path(".").rglob("__pycache__"):
        if pycache_dir.is_dir():
            for file_path in pycache_dir.rglob("*"):
                if file_path.is_file():
                    redundant_files.append((file_path, "Cache Python (INNECESARIO)"))
    
    # 4. Checkpoints antiguos (opcional)
    checkpoint_dir = Path("evaluation/.langgraph_checkpoints")
    if checkpoint_dir.exists():
        # Solo sugerir checkpoints antiguos (más de 7 días)
        import time
        week_ago = time.time() - (7 * 24 * 60 * 60)
        
        for file_path in checkpoint_dir.rglob("*"):
            if file_path.is_file() and file_path.stat().st_mtime < week_ago:
                redundant_files.append((file_path, "Checkpoint antiguo (>7 días)"))
    
    return redundant_files

def calculate_total_size(file_paths: List[Path]) -> int:
    """Calcular el tamaño total de archivos en bytes."""
    total_size = 0
    for file_path in file_paths:
        try:
            if file_path.exists() and file_path.is_file():
                total_size += file_path.stat().st_size
        except (OSError, PermissionError):
            continue
    return total_size

def format_size(size_bytes: int) -> str:
    """Formatear tamaño en bytes a formato legible."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def cleanup_redundant_files(dry_run: bool = False) -> None:
    """
    Limpia archivos redundantes del proyecto.
    
    Args:
        dry_run: Si True, solo muestra qué se eliminaría sin hacer cambios
    """
    print("🧹 LIMPIEZA DE ARCHIVOS REDUNDANTES")
    print("=" * 60)
    
    redundant_files = find_redundant_files()
    
    if not redundant_files:
        print("✅ No se encontraron archivos redundantes")
        return
    
    # Agrupar por tipo
    grouped_files = {}
    for file_path, description in redundant_files:
        category = description.split("(")[1].rstrip(")")
        if category not in grouped_files:
            grouped_files[category] = []
        grouped_files[category].append(file_path)
    
    # Mostrar resumen
    total_files = len(redundant_files)
    total_size = calculate_total_size([fp for fp, _ in redundant_files])
    
    print(f"📊 Archivos encontrados: {total_files}")
    print(f"💾 Espacio total: {format_size(total_size)}")
    
    # Mostrar por categoría
    for category, files in grouped_files.items():
        category_size = calculate_total_size(files)
        print(f"\n📁 {category}: {len(files)} archivos ({format_size(category_size)})")
        
        # Mostrar algunos ejemplos
        for i, file_path in enumerate(files[:3]):
            try:
                # Usar ruta absoluta para evitar errores
                if file_path.is_absolute():
                    display_path = str(file_path)
                else:
                    display_path = str(file_path)
                file_size = format_size(file_path.stat().st_size) if file_path.exists() else "0 B"
                print(f"   • {display_path} ({file_size})")
            except (ValueError, OSError):
                print(f"   • {str(file_path)} (error de acceso)")
        
        if len(files) > 3:
            print(f"   ... y {len(files) - 3} archivos más")
    
    if dry_run:
        print(f"\n🔍 MODO DRY-RUN: No se eliminó nada")
        print(f"📋 Para eliminar realmente, ejecuta sin --dry-run")
        return
    
    # Confirmar eliminación
    print(f"\n⚠️  ¿ELIMINAR {total_files} archivos ({format_size(total_size)})?")
    response = input("Escriba 'confirmar' para continuar: ")
    
    if response.lower() != 'confirmar':
        print("❌ Operación cancelada")
        return
    
    # Eliminar archivos
    deleted_count = 0
    errors = []
    
    print(f"\n🗑️  Eliminando archivos...")
    
    for file_path, description in redundant_files:
        try:
            if file_path.exists():
                if file_path.is_file():
                    file_path.unlink()
                    deleted_count += 1
                elif file_path.is_dir() and not any(file_path.iterdir()):
                    file_path.rmdir()
                    deleted_count += 1
        except (OSError, PermissionError) as e:
            errors.append(f"{file_path}: {e}")
    
    # Eliminar directorios vacíos
    empty_dirs = []
    for category, files in grouped_files.items():
        for file_path in files:
            parent = file_path.parent
            try:
                if parent.exists() and parent.is_dir() and not any(parent.iterdir()):
                    parent.rmdir()
                    empty_dirs.append(parent)
            except (OSError, PermissionError):
                continue
    
    # Resumen final
    print(f"✅ Eliminados: {deleted_count} archivos")
    if empty_dirs:
        print(f"📁 Directorios vacíos eliminados: {len(empty_dirs)}")
    
    if errors:
        print(f"\n⚠️  Errores ({len(errors)}):")
        for error in errors[:5]:
            print(f"   • {error}")
        if len(errors) > 5:
            print(f"   ... y {len(errors) - 5} errores más")
    
    freed_space = calculate_total_size([fp for fp, _ in redundant_files if not fp.exists()])
    print(f"\n💾 Espacio liberado: {format_size(total_size - freed_space)}")

def main():
    """Función principal"""
    parser = argparse.ArgumentParser(description="Limpiar archivos redundantes del proyecto")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Solo mostrar qué se eliminaría, sin hacer cambios")
    
    args = parser.parse_args()
    
    try:
        cleanup_redundant_files(dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\n❌ Operación cancelada por el usuario")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    main() 
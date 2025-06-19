#!/usr/bin/env python3
"""
Script de Gestión de Checkpoints

Herramientas para manejar checkpoints de evaluaciones:
- Listar checkpoints disponibles
- Ver detalles de un checkpoint
- Limpiar checkpoints antiguos
- Reanudar evaluaciones desde checkpoint

Uso:
    python -m evaluation.scripts.manage_checkpoints --list
    python -m evaluation.scripts.manage_checkpoints --details checkpoint_id
    python -m evaluation.scripts.manage_checkpoints --cleanup
"""

import argparse
import json

from evaluation.core.network_resilience import EvaluationCheckpoint


def list_checkpoints():
    """Listar todos los checkpoints disponibles"""
    checkpoint_system = EvaluationCheckpoint()
    checkpoint_dir = checkpoint_system.checkpoint_dir
    
    if not checkpoint_dir.exists():
        print("📁 No se encontró directorio de checkpoints")
        return
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
    
    if not checkpoints:
        print("📁 No se encontraron checkpoints")
        return
    
    print(f"📁 Checkpoints disponibles ({len(checkpoints)}):")
    print("-" * 80)
    
    # Agrupar por evaluation_id
    checkpoint_groups = {}
    for checkpoint_file in checkpoints:
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            eval_id = data.get("evaluation_id", "unknown")
            if eval_id not in checkpoint_groups:
                checkpoint_groups[eval_id] = []
            
            checkpoint_groups[eval_id].append({
                "file": checkpoint_file,
                "data": data
            })
        except Exception as e:
            print(f"⚠️  Error leyendo {checkpoint_file}: {e}")
    
    for eval_id, checkpoints_list in checkpoint_groups.items():
        print(f"\n🔍 Evaluación: {eval_id}")
        
        # Ordenar por timestamp
        checkpoints_list.sort(key=lambda x: x["data"].get("timestamp", ""))
        
        for checkpoint in checkpoints_list:
            file_path = checkpoint["file"]
            data = checkpoint["data"]
            timestamp = data.get("timestamp", "unknown")
            
            # Información del contenido
            if "data" in data and isinstance(data["data"], dict):
                content = data["data"]
                agent_name = content.get("agent_name", "unknown")
                completed = content.get("completed_samples", 0)
                total = content.get("total_samples", 0)
                status = content.get("status", "in_progress")
                
                status_icon = "✅" if status == "completed" else "🔄"
                
                print(f"  {status_icon} {file_path.name}")
                print(f"     📅 {timestamp}")
                print(f"     🤖 Agente: {agent_name}")
                print(f"     📊 Progreso: {completed}/{total} ({completed/total*100 if total > 0 else 0:.1f}%)")
            else:
                print(f"  📄 {file_path.name}")
                print(f"     📅 {timestamp}")


def show_checkpoint_details(checkpoint_id: str):
    """Mostrar detalles de un checkpoint específico"""
    checkpoint_system = EvaluationCheckpoint()
    
    data = checkpoint_system.load_latest_checkpoint(checkpoint_id)
    
    if not data:
        print(f"❌ No se encontró checkpoint para: {checkpoint_id}")
        return
    
    print(f"🔍 Detalles del checkpoint: {checkpoint_id}")
    print("=" * 60)
    
    # Información general
    print(f"📅 Timestamp: {data.get('timestamp', 'unknown')}")
    print(f"🆔 Evaluation ID: {data.get('evaluation_id', 'unknown')}")
    
    # Información del contenido
    if "data" in data and isinstance(data["data"], dict):
        content = data["data"]
        
        print(f"🤖 Agente: {content.get('agent_name', 'unknown')}")
        print(f"📊 Muestras totales: {content.get('total_samples', 0)}")
        print(f"✅ Muestras completadas: {content.get('completed_samples', 0)}")
        print(f"📈 Estado: {content.get('status', 'in_progress')}")
        
        # Información de error si existe
        if "last_error" in content:
            print(f"❌ Último error: {content['last_error']}")
            print(f"🔢 Muestra fallida: {content.get('failed_sample_index', 'unknown')}")
        
        # Estadísticas de resultados
        results = content.get("results", [])
        if results:
            successful = len([r for r in results if r.get("technical_success", False)])
            quality_scores = [r.get("quality_score", 0) for r in results if r.get("quality_score") is not None]
            
            print("\n📊 Estadísticas de resultados:")
            print(f"   • Total de resultados: {len(results)}")
            print(f"   • Éxitos técnicos: {successful} ({successful/len(results)*100:.1f}%)")
            if quality_scores:
                avg_quality = sum(quality_scores) / len(quality_scores)
                print(f"   • Calidad promedio: {avg_quality:.3f}")
    
    print("\n💡 Para reanudar esta evaluación usa:")
    print(f"   python -m evaluation.scripts.run_evaluation_local --resume-from {checkpoint_id}")


def cleanup_checkpoints(keep_last: int = 3, dry_run: bool = False):
    """Limpiar checkpoints antiguos"""
    checkpoint_system = EvaluationCheckpoint()
    checkpoint_dir = checkpoint_system.checkpoint_dir
    
    if not checkpoint_dir.exists():
        print("📁 No se encontró directorio de checkpoints")
        return
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
    
    if not checkpoints:
        print("📁 No se encontraron checkpoints")
        return
    
    # Agrupar por evaluation_id
    checkpoint_groups = {}
    for checkpoint_file in checkpoints:
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            eval_id = data.get("evaluation_id", "unknown")
            
            if eval_id not in checkpoint_groups:
                checkpoint_groups[eval_id] = []
            
            checkpoint_groups[eval_id].append(checkpoint_file)
        except:
            continue
    
    total_deleted = 0
    
    for eval_id, files in checkpoint_groups.items():
        if len(files) <= keep_last:
            continue
        
        # Ordenar por tiempo de modificación y eliminar los más antiguos
        files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        to_delete = files[keep_last:]
        
        print(f"\n🗑️  Limpiando checkpoints para {eval_id}:")
        print(f"   Manteniendo: {keep_last} más recientes")
        print(f"   Eliminando: {len(to_delete)} antiguos")
        
        for file_path in to_delete:
            if dry_run:
                print(f"   🔍 [DRY-RUN] Eliminaría: {file_path.name}")
            else:
                try:
                    file_path.unlink()
                    print(f"   ✅ Eliminado: {file_path.name}")
                    total_deleted += 1
                except Exception as e:
                    print(f"   ❌ Error eliminando {file_path.name}: {e}")
    
    if dry_run:
        print(f"\n🔍 [DRY-RUN] Se eliminarían {total_deleted} checkpoints")
        print("💡 Usa --confirm para eliminar realmente")
    else:
        print(f"\n✅ Limpieza completada: {total_deleted} checkpoints eliminados")


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Gestión de Checkpoints de Evaluación",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Listar todos los checkpoints
  python -m evaluation.scripts.manage_checkpoints --list
  
  # Ver detalles de un checkpoint específico
  python -m evaluation.scripts.manage_checkpoints --details my_evaluation_id
  
  # Limpiar checkpoints antiguos (dry-run)
  python -m evaluation.scripts.manage_checkpoints --cleanup
  
  # Limpiar checkpoints antiguos (confirmar)
  python -m evaluation.scripts.manage_checkpoints --cleanup --confirm
        """
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="Listar todos los checkpoints disponibles"
    )
    
    parser.add_argument(
        "--details",
        type=str,
        help="Mostrar detalles de un checkpoint específico"
    )
    
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Limpiar checkpoints antiguos"
    )
    
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Número de checkpoints a mantener por evaluación (default: 3)"
    )
    
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirmar eliminación de checkpoints (sin esto es dry-run)"
    )
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints()
    elif args.details:
        show_checkpoint_details(args.details)
    elif args.cleanup:
        cleanup_checkpoints(keep_last=args.keep, dry_run=not args.confirm)
    else:
        parser.print_help()


if __name__ == "__main__":
    main() 
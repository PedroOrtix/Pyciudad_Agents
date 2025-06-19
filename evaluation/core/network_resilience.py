"""
Sistema de Robustez de Red para Evaluaciones PyCiudad

Este módulo proporciona herramientas para hacer que las evaluaciones largas
sean resilientes ante fallos de conectividad de red:

- Retry automático con backoff exponencial
- Detección inteligente de errores de conectividad
- Pausa automática y espera por reconexión
- Sistema de guardar/reanudar evaluaciones
- Logging detallado de errores de red

Uso:
    @with_network_resilience()
    def mi_funcion_con_red():
        # código que hace llamadas de red
        pass
"""

import asyncio
import time
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Callable, Dict, Optional
import functools
import socket
import requests
from requests.exceptions import (
    ConnectionError, 
    Timeout, 
    RequestException
)
from httpx import ConnectError, TimeoutException

# Configurar logging específico para problemas de red
logger = logging.getLogger("network_resilience")
logger.setLevel(logging.INFO)

# Handler para archivo de log
log_handler = logging.FileHandler("evaluation/network_resilience.log")
log_formatter = logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
)
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

# También log a consola
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)


class NetworkError(Exception):
    """Excepción para errores de red detectados"""
    pass


class ConnectivityChecker:
    """Verificador de conectividad de red"""
    
    def __init__(self):
        self.test_urls = [
            "https://www.google.com",
            "https://httpbin.org/get",
            "https://1.1.1.1"  # Cloudflare DNS
        ]
        self.timeout = 5
    
    def is_internet_available(self) -> bool:
        """Verificar si hay conexión a internet"""
        for url in self.test_urls:
            try:
                response = requests.get(url, timeout=self.timeout)
                if response.status_code == 200:
                    return True
            except:
                continue
        return False
    
    def check_specific_host(self, host: str, port: int = 80) -> bool:
        """Verificar conectividad a un host específico"""
        try:
            socket.create_connection((host, port), timeout=self.timeout)
            return True
        except (socket.error, socket.timeout):
            return False
    
    def wait_for_connectivity(self, max_wait_minutes: int = 30) -> bool:
        """Esperar hasta que regrese la conectividad"""
        start_time = time.time()
        max_wait_seconds = max_wait_minutes * 60
        
        logger.info(f"🔄 Esperando conectividad (máximo {max_wait_minutes} minutos)...")
        
        while time.time() - start_time < max_wait_seconds:
            if self.is_internet_available():
                logger.info("✅ Conectividad restaurada!")
                return True
            
            # Mostrar progreso cada 30 segundos
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:
                remaining = max_wait_seconds - elapsed
                logger.info(f"⏳ Esperando conectividad... {remaining/60:.1f} min restantes")
            
            time.sleep(10)  # Verificar cada 10 segundos
        
        logger.error(f"❌ Timeout esperando conectividad ({max_wait_minutes} min)")
        return False


class NetworkResilientWrapper:
    """Wrapper para hacer funciones resilientes ante fallos de red"""
    
    def __init__(self, 
                 max_retries: int = 5,
                 base_delay: float = 1.0,
                 max_delay: float = 300.0,  # 5 minutos
                 exponential_base: float = 2.0,
                 connectivity_wait_minutes: int = 30):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
        self.connectivity_wait_minutes = connectivity_wait_minutes
        self.connectivity_checker = ConnectivityChecker()
        
    def is_network_error(self, exception: Exception) -> bool:
        """Detectar si una excepción es un error de red"""
        network_exceptions = (
            ConnectionError,
            Timeout,
            RequestException,
            ConnectError,
            TimeoutException,
            socket.error,
            socket.timeout,
            OSError  # Para errores de SO relacionados con red
        )
        
        # Verificar tipo de excepción
        if isinstance(exception, network_exceptions):
            return True
        
        # Verificar mensajes específicos
        error_msg = str(exception).lower()
        network_indicators = [
            "connection",
            "timeout",
            "network",
            "unreachable",
            "failed to connect",
            "connection refused",
            "name resolution failed",
            "temporary failure in name resolution",
            "no route to host",
            "connection reset"
        ]
        
        return any(indicator in error_msg for indicator in network_indicators)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calcular delay para retry con backoff exponencial"""
        delay = self.base_delay * (self.exponential_base ** attempt)
        # Añadir jitter (variación aleatoria) para evitar thundering herd
        import random
        jitter = random.uniform(0.8, 1.2)
        delay *= jitter
        return min(delay, self.max_delay)
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator para hacer una función resiliente"""
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if not self.is_network_error(e):
                        # No es error de red, propagar inmediatamente
                        raise e
                    
                    logger.warning(f"🔄 Error de red en intento {attempt + 1}/{self.max_retries + 1}: {e}")
                    
                    if attempt == self.max_retries:
                        # Último intento fallido
                        logger.error(f"❌ Máximo número de reintentos alcanzado ({self.max_retries})")
                        break
                    
                    # Verificar conectividad
                    if not self.connectivity_checker.is_internet_available():
                        logger.warning("🌐 Sin conectividad detectada, esperando...")
                        if not self.connectivity_checker.wait_for_connectivity(self.connectivity_wait_minutes):
                            raise NetworkError(f"Sin conectividad después de {self.connectivity_wait_minutes} minutos")
                    
                    # Esperar antes del siguiente intento
                    delay = self.calculate_delay(attempt)
                    logger.info(f"⏳ Esperando {delay:.1f}s antes del siguiente intento...")
                    time.sleep(delay)
            
            # Si llegamos aquí, todos los intentos fallaron
            raise NetworkError(f"Falló después de {self.max_retries + 1} intentos. Último error: {last_exception}")
        
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(self.max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                    
                except Exception as e:
                    last_exception = e
                    
                    if not self.is_network_error(e):
                        # No es error de red, propagar inmediatamente
                        raise e
                    
                    logger.warning(f"🔄 Error de red en intento {attempt + 1}/{self.max_retries + 1}: {e}")
                    
                    if attempt == self.max_retries:
                        # Último intento fallido
                        logger.error(f"❌ Máximo número de reintentos alcanzado ({self.max_retries})")
                        break
                    
                    # Verificar conectividad
                    if not self.connectivity_checker.is_internet_available():
                        logger.warning("🌐 Sin conectividad detectada, esperando...")
                        if not self.connectivity_checker.wait_for_connectivity(self.connectivity_wait_minutes):
                            raise NetworkError(f"Sin conectividad después de {self.connectivity_wait_minutes} minutos")
                    
                    # Esperar antes del siguiente intento
                    delay = self.calculate_delay(attempt)
                    logger.info(f"⏳ Esperando {delay:.1f}s antes del siguiente intento...")
                    await asyncio.sleep(delay)
            
            # Si llegamos aquí, todos los intentos fallaron
            raise NetworkError(f"Falló después de {self.max_retries + 1} intentos. Último error: {last_exception}")
        
        # Detectar si la función es async o sync
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper


class EvaluationCheckpoint:
    """Sistema de checkpoint para reanudar evaluaciones"""
    
    def __init__(self, checkpoint_dir: str = "evaluation/checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, evaluation_id: str, data: Dict[str, Any]) -> str:
        """Guardar checkpoint de evaluación"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{evaluation_id}_{timestamp}.json"
        filepath = self.checkpoint_dir / filename
        
        checkpoint_data = {
            "evaluation_id": evaluation_id,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"💾 Checkpoint guardado: {filepath}")
        return str(filepath)
    
    def load_latest_checkpoint(self, evaluation_id: str) -> Optional[Dict[str, Any]]:
        """Cargar el checkpoint más reciente para una evaluación"""
        pattern = f"checkpoint_{evaluation_id}_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            return None
        
        # Obtener el más reciente por timestamp
        latest = max(checkpoints, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"📂 Checkpoint cargado: {latest}")
            return data
        except Exception as e:
            logger.error(f"❌ Error cargando checkpoint {latest}: {e}")
            return None
    
    def cleanup_old_checkpoints(self, evaluation_id: str, keep_last: int = 3):
        """Limpiar checkpoints antiguos"""
        pattern = f"checkpoint_{evaluation_id}_*.json"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) <= keep_last:
            return
        
        # Ordenar por tiempo de modificación y eliminar los más antiguos
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        to_delete = checkpoints[keep_last:]
        
        for checkpoint in to_delete:
            try:
                checkpoint.unlink()
                logger.info(f"🗑️  Checkpoint eliminado: {checkpoint}")
            except Exception as e:
                logger.warning(f"⚠️  No se pudo eliminar {checkpoint}: {e}")


# Decoradores pre-configurados comunes
def with_network_resilience(max_retries: int = 5, 
                          base_delay: float = 1.0,
                          max_delay: float = 300.0,
                          connectivity_wait_minutes: int = 30):
    """Decorator para hacer funciones resilientes ante fallos de red"""
    return NetworkResilientWrapper(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        connectivity_wait_minutes=connectivity_wait_minutes
    )


def with_quick_retry(max_retries: int = 3, base_delay: float = 0.5):
    """Decorator para reintentos rápidos (para operaciones que deberían ser inmediatas)"""
    return NetworkResilientWrapper(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=30.0,
        connectivity_wait_minutes=5
    )


def with_patient_retry(max_retries: int = 10, connectivity_wait_minutes: int = 60):
    """Decorator para reintentos pacientes (para evaluaciones largas)"""
    return NetworkResilientWrapper(
        max_retries=max_retries,
        base_delay=2.0,
        max_delay=600.0,  # 10 minutos
        connectivity_wait_minutes=connectivity_wait_minutes
    )


# Función de utilidad para testear conectividad
def test_connectivity():
    """Función para testear el sistema de conectividad"""
    checker = ConnectivityChecker()
    
    print("🔍 Testando conectividad...")
    print(f"Internet disponible: {checker.is_internet_available()}")
    print(f"Google alcanzable: {checker.check_specific_host('google.com', 80)}")
    print(f"Localhost:11434 (Ollama): {checker.check_specific_host('localhost', 11434)}")


if __name__ == "__main__":
    # Test del sistema
    test_connectivity() 
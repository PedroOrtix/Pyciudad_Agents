import os
from langchain_ollama import ChatOllama
from evaluation.core.network_resilience import with_network_resilience

# Initialize LLM models
model_name = os.environ.get("OLLAMA_MODEL", "qwen3:30b-a3b") # Default model
model_name_thinking = os.environ.get("OLLAMA_MODEL_THINKING", "qwen3:30b-a3b") # Default model for thinking
base_url = os.environ.get("OLLAMA_HOST_PORT", "http://localhost:11434") # Default host

# Configurar timeout más largo para manejar modelos grandes
REQUEST_TIMEOUT = 120  # 2 minutos

if "qwen3" in model_name_thinking:
    llm_base = ChatOllama(model=model_name, base_url=base_url, timeout=REQUEST_TIMEOUT)
    llm_thinking_base = ChatOllama(model=model_name_thinking, base_url=base_url, think=True, timeout=REQUEST_TIMEOUT)
else:
    llm_base = ChatOllama(model=model_name, base_url=base_url, timeout=REQUEST_TIMEOUT)
    llm_thinking_base = ChatOllama(model=model_name_thinking, base_url=base_url, timeout=REQUEST_TIMEOUT)


class ResilientLLM:
    """Wrapper robusto para LLMs que maneja errores de conectividad"""
    
    def __init__(self, base_llm, name: str):
        self.base_llm = base_llm
        self.name = name
    
    @with_network_resilience(max_retries=5, base_delay=3.0, connectivity_wait_minutes=15)
    def invoke(self, *args, **kwargs):
        """Invoke robusto con reintentos automáticos"""
        try:
            return self.base_llm.invoke(*args, **kwargs)
        except Exception as e:
            print(f"🔄 Error en LLM {self.name}: {e}")
            raise
    
    @with_network_resilience(max_retries=5, base_delay=3.0, connectivity_wait_minutes=15)
    async def ainvoke(self, *args, **kwargs):
        """Async invoke robusto con reintentos automáticos"""
        try:
            return await self.base_llm.ainvoke(*args, **kwargs)
        except Exception as e:
            print(f"🔄 Error en LLM {self.name}: {e}")
            raise
    
    def with_structured_output(self, schema):
        """Crear una versión con salida estructurada que también sea resiliente"""
        structured_llm = self.base_llm.with_structured_output(schema)
        return ResilientStructuredLLM(structured_llm, f"{self.name}_structured")
    
    def __getattr__(self, name):
        """Delegar otros métodos al LLM base"""
        return getattr(self.base_llm, name)


class ResilientStructuredLLM:
    """Wrapper robusto para LLMs con salida estructurada"""
    
    def __init__(self, structured_llm, name: str):
        self.structured_llm = structured_llm
        self.name = name
    
    @with_network_resilience(max_retries=5, base_delay=3.0, connectivity_wait_minutes=15)
    def invoke(self, *args, **kwargs):
        """Invoke robusto para LLM estructurado"""
        try:
            return self.structured_llm.invoke(*args, **kwargs)
        except Exception as e:
            print(f"🔄 Error en LLM estructurado {self.name}: {e}")
            raise
    
    @with_network_resilience(max_retries=5, base_delay=3.0, connectivity_wait_minutes=15)
    async def ainvoke(self, *args, **kwargs):
        """Async invoke robusto para LLM estructurado"""
        try:
            return await self.structured_llm.ainvoke(*args, **kwargs)
        except Exception as e:
            print(f"🔄 Error en LLM estructurado {self.name}: {e}")
            raise
    
    def __getattr__(self, name):
        """Delegar otros métodos al LLM base"""
        return getattr(self.structured_llm, name)


# Crear instancias robustas
llm = ResilientLLM(llm_base, "ollama_standard")
llm_thinking = ResilientLLM(llm_thinking_base, "ollama_thinking")
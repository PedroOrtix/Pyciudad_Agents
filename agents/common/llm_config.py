import os
from langchain_ollama import ChatOllama

# Initialize LLM models
model_name = os.environ.get("OLLAMA_MODEL", "qwen3:30b-a3b") # Default model
model_name_thinking = os.environ.get("OLLAMA_MODEL_THINKING", "qwq:32b") # Default model for thinking
base_url = os.environ.get("OLLAMA_HOST_PORT", "http://localhost:11434") # Default host

llm = ChatOllama(model=model_name, base_url=base_url)
llm_thinking = ChatOllama(model=model_name_thinking, base_url=base_url)
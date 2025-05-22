import os
from langchain_ollama import ChatOllama

# Initialize LLM models
model_name = os.environ.get("OLLAMA_MODEL", "llama3:instruct") # Default model
base_url = os.environ.get("OLLAMA_HOST_PORT", "http://localhost:11434") # Default host

llm = ChatOllama(model=model_name, base_url=base_url)
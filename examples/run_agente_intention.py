import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.Agent_intention.agent_intention import app_intention

# Puedes definir aquí la consulta para pruebas directas
query = ""

if __name__ == "__main__":
    consulta = query.strip() if query.strip() else (sys.argv[1] if len(sys.argv) > 1 else None)
    if not consulta:
        print("Uso: python run_agente_intention.py 'consulta de ejemplo'\nO define la variable 'query' en el script.")
        sys.exit(1)
    result = app_intention.invoke({"user_query": consulta})
    print("\n--- Resultados Agente Intention ---")
    for cand in result.get("final_candidates", []):
        print(f"- {cand}") 
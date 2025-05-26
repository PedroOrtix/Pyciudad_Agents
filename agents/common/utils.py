# Utilidades comunes para los agentes

def deduplicate_candidates(candidates):
    """Elimina duplicados de una lista de candidatos usando 'address' como clave principal y 'id' como respaldo."""
    unique = {}
    for cand in candidates:
        if hasattr(cand, 'address') and cand.address:
            if cand.address not in unique:
                unique[cand.address] = cand
        elif hasattr(cand, 'id') and cand.id:
            if cand.id not in unique:
                unique[cand.id] = cand
    return list(unique.values())

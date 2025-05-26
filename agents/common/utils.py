# Utilidades comunes para los agentes

from typing import List
from agents.common.schemas import CandidateSchema

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

def reorder_candidates_by_ids(ordered_ids: List[str], original_candidates: List[CandidateSchema]) -> List[CandidateSchema]:
    """
    Reordena la lista de candidatos original según el orden de IDs proporcionado.
    """
    candidates_dict = {c.id: c for c in original_candidates if c.id is not None}
    return [candidates_dict[cid] for cid in ordered_ids if cid in candidates_dict]

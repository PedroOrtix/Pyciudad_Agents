# Utilidades comunes para los agentes

from typing import List, Dict, Any
from agents.common.schemas import CandidateSchema

def deduplicate_candidates(candidates: List[CandidateSchema]) -> List[CandidateSchema]:
    """
    Elimina candidatos duplicados de una lista usando 'address' como clave principal.
    
    Si un candidato no tiene dirección, usa 'id' como clave de respaldo.
    
    Args:
        candidates: Lista de candidatos a deduplicar
        
    Returns:
        Lista de candidatos únicos
    """
    unique: Dict[str, CandidateSchema] = {}
    
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
    Reordena una lista de candidatos según el orden de IDs especificado.
    
    Args:
        ordered_ids: Lista de IDs en el orden deseado
        original_candidates: Lista original de candidatos
        
    Returns:
        Lista de candidatos reordenados según los IDs proporcionados.
        Solo incluye candidatos cuyos IDs estén en ordered_ids.
    """
    candidates_dict = {c.id: c for c in original_candidates if c.id is not None}
    return [candidates_dict[cid] for cid in ordered_ids if cid in candidates_dict]

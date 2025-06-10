from typing import List, Optional, Union, Dict, Any
from pyciudad.cliente import CartoCiudad
from evaluation.core.network_resilience import with_network_resilience

# Initialize CartoCiudad client once here
cartociudad_client = CartoCiudad(debug=False)

@with_network_resilience(max_retries=5, base_delay=2.0, connectivity_wait_minutes=10)
def search_cartociudad_tool(
    consulta: str,
    limite: int = 10,
    excluir_tipos: Optional[List[str]] = None,
    codigo_postal: Optional[Union[str, List[str]]] = None,
    municipio: Optional[Union[str, List[str]]] = None,
    provincia: Optional[Union[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Busca candidatos en CartoCiudad que coincidan con la consulta.
    Ahora con robustez de red: reintentos automáticos ante fallos de conectividad.
    """
    api_results = cartociudad_client.buscar_candidatos(
        consulta=consulta,
        limite=limite,
        excluir_tipos=excluir_tipos,
        codigo_postal=codigo_postal,
        municipio=municipio,
        provincia=provincia,
    )
    return [candidate.model_dump() for candidate in api_results]
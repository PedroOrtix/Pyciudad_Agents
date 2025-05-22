from typing import List, Optional, Union, Dict, Any
from pyciudad.cliente import CartoCiudad

# Initialize CartoCiudad client once here
cartociudad_client = CartoCiudad(debug=False)

def search_cartociudad_tool(
    consulta: str,
    limite: int = 10,
    excluir_tipos: Optional[List[str]] = None,
    codigo_postal: Optional[Union[str, List[str]]] = None,
    municipio: Optional[Union[str, List[str]]] = None,
    provincia: Optional[Union[str, List[str]]] = None,
) -> List[Dict[str, Any]]:
    """
    Search for candidates in CartoCiudad that match the query.
    This is the centralized tool used by all agents.
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
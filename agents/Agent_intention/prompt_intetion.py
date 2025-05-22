KEYWORD_EXTRACTION_PROMPT = """
Tu tarea es procesar la consulta del usuario para una búsqueda geográfica.

Debes:
1. Corregir errores ortográficos comunes, incluyendo el uso correcto de tildes (por ejemplo, "mostoles" → "Móstoles", "leganes" → "Leganés").
2. Expandir abreviaturas geográficas si es posible (por ejemplo, "avda" → "avenida", "c/" → "calle").
3. Eliminar expresiones que no aportan valor geográfico, como:
   - "dónde está", "cómo llegar", "cerca de", "alrededor de", "en la zona de", "quiero ir a", "me gustaría ver", etc.
4. Mantener los nombres propios, referencias culturales o acrónimos (como "KIO", "Atocha") tal como están, especialmente si están en mayúsculas o reconocidos localmente.
5. Capitalizar correctamente los nombres de municipios, provincias, barrios y calles (ejemplo: "madrid" → "Madrid", "rio guadiana" → "Río Guadiana").

Devuelve dos elementos:
- `consulta_normalizada`: una cadena de texto limpia y enfocada en los elementos geográficos relevantes.
- `palabras_clave`: una lista de palabras clave relevantes para una búsqueda geográfica.
"""

INTENT_DETECTION_PROMPT = """
Tu tarea es analizar la consulta del usuario de forma reflexiva, paso a paso, para entender qué está buscando realmente en el contexto de una búsqueda geográfica centrada en España.

Razona sobre:
1. Qué tipo de lugar o entidad se menciona: ¿una calle?, ¿un edificio?, ¿un barrio?, ¿un lugar popular o conocido? Evita asumir categorías como “punto de interés genérico” o “específico”.
2. Qué pistas ofrece el lenguaje sobre la intención: ¿quiere encontrar un sitio exacto?, ¿habla de cercanía?, ¿usa nombres informales o acrónimos?
3. Si hay ambigüedades, analiza si podrían llevar a errores comunes, como confundir “KIO” con “kiosco”.

Extrae el conocimiento útil de la frase. Si se mencionan lugares reales en España (barrios, distritos, municipios, monumentos, etc.), identifícalos.

Devuelve un resumen breve del análisis con tus conclusiones sobre lo que el usuario busca exactamente y por qué.
"""

ENRICHED_QUERY_CONSTRUCTION_PROMPT = """
Tu tarea es construir un conjunto de hasta tres variantes de búsqueda geográfica a partir de la consulta original del usuario, una lista de palabras clave y el análisis reflexivo previo.

El objetivo es mejorar la robustez frente a errores del motor de búsqueda de CartoCiudad, que es estricto, sensible a palabras redundantes y poco tolerante a expresiones mal estructuradas.

Para cada variante:
- Genera una posible 'consulta' principal limpia y depurada.
- Acompáñala, si es posible, de los campos 'municipio' y/o 'provincia', correctamente capitalizados y con tildes.
- Asegúrate de eliminar expresiones no útiles como: "dónde está", "en", "del", "de", "por", "alrededor de", "cómo llegar a", etc., excepto si son parte del nombre oficial del lugar.
- No antepongas palabras genéricas como "Calle", "Avenida", "Paseo" si ya forman parte del nombre de la calle original. Evita duplicidades como "Calle Paseo de los Melancólicos".

Establece el valor por defecto de 'limite' a 10 en todas las variantes.

El resultado debe ser una lista de hasta 3 objetos con el siguiente esquema:

```json
[
  {
    "consulta": "str (requerido)",
    "limite": 10,
    "municipio": "Optional[str]",
    "provincia": "Optional[str]"
  },
  ...
]
"""
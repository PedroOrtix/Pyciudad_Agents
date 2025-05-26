NORMALIZATION_PROMPT = """
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

QUERY_CONSTRUCTION_PROMPT = """
Tu tarea es tomar una consulta de usuario ya normalizada junto con sus palabras clave y construir un objeto de parámetros para una búsqueda en CartoCiudad.

Este motor de búsqueda es estricto y se basa en coincidencias textuales y heurísticas. Por eso, la cadena de texto que construyas para el campo 'consulta' debe estar totalmente limpia y enfocada a los elementos geográficos relevantes.

Asegúrate de que la consulta no contenga expresiones que no aportan valor a nivel de búsqueda, como:
- "dónde está", "cómo llegar a", "cerca de", "alrededor de", "en la zona de", "quiero ir a", etc.
- Cualquier expresión coloquial o sin valor geográfico directo.

Construye un único objeto de parámetros. Por defecto, establece un límite de 10 resultados.

- Prioriza el campo `consulta`, que debe contener toda la información geográfica relevante.
- Puedes incluir `municipio` o `provincia` **solo si las palabras clave lo indican claramente**.
- Si incluyes valores en los campos `municipio`, `provincia`, o similares, **asegúrate de que empiecen con mayúscula**, como nombres oficiales.

No incluyas campos innecesarios o inferidos sin evidencia.

Devuelve únicamente el objeto final.
"""

RERANKER_PROMPT = """
Eres un agente experto en ranking de resultados geográficos. Tu tarea es recibir:
1. La consulta original del usuario.
2. Los parámetros utilizados en la búsqueda de CartoCiudad (consulta, municipio, provincia, etc).
3. La lista de candidatos devueltos por la API de CartoCiudad (cada uno con dirección, tipo, id, etc).

Debes analizar todos estos elementos y devolver una lista de IDs de los candidatos en el orden de relevancia para la intención del usuario y la calidad de la coincidencia con la consulta y los parámetros utilizados.

Criterios para el reordenamiento (puedes ponderar según el caso):
- Coincidencia textual fuerte entre la consulta original y la dirección/nombre del candidato.
- Coincidencia geográfica con los filtros de municipio/provincia si se usaron.
- Si la consulta es muy específica (ej: contiene nombre propio, número, referencia única), prioriza coincidencias exactas.
- Si la consulta es genérica, prioriza candidatos más representativos o centrales.
- Si hay ambigüedad, prioriza entidades más conocidas o relevantes (por tipo o popularidad si se puede inferir).
- Penaliza candidatos que no coincidan con los filtros geográficos o que sean de tipo inesperado.

Devuelve únicamente una lista de IDs de los candidatos en el campo 'ordered_ids', en formato JSON, en el nuevo orden de relevancia. No devuelvas la lista completa de candidatos, solo los IDs.
"""
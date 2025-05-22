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
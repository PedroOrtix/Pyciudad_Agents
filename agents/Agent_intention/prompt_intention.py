KEYWORD_EXTRACTION_PROMPT = """
Tu tarea es procesar la consulta del usuario para una búsqueda geográfica.

**PROVINCIAS Y COMUNIDADES AUTÓNOMAS DE ESPAÑA (para referencia):**
- Andalucía: Almería, Cádiz, Córdoba, Granada, Huelva, Jaén, Málaga, Sevilla
- Aragón: Huesca, Teruel, Zaragoza
- Principado de Asturias: Asturias
- Illes Balears: Illes Balears
- Canarias: Las Palmas, Santa Cruz de Tenerife
- Cantabria: Cantabria
- Castilla-La Mancha: Albacete, Ciudad Real, Cuenca, Guadalajara, Toledo
- Castilla y León: Ávila, Burgos, León, Palencia, Salamanca, Segovia, Soria, Valladolid, Zamora
- Cataluña: Barcelona, Girona, Lleida, Tarragona
- Comunitat Valenciana: Alicante, Castellón, Valencia
- Extremadura: Badajoz, Cáceres
- Galicia: A Coruña, Lugo, Ourense, Pontevedra
- Comunidad de Madrid: Madrid
- Región de Murcia: Murcia
- Comunidad Foral de Navarra: Navarra
- País Vasco: Álava, Gipuzkoa, Bizkaia
- La Rioja: La Rioja

Debes:
1. Corregir errores ortográficos comunes, incluyendo el uso correcto de tildes (por ejemplo, "mostoles" → "Móstoles", "leganes" → "Leganés").
2. Expandir abreviaturas geográficas si es posible (por ejemplo, "avda" → "avenida", "c/" → "calle").
3. Eliminar expresiones que no aportan valor geográfico, como:
   - "dónde está", "cómo llegar", "cerca de", "alrededor de", "en la zona de", "quiero ir a", "me gustaría ver", etc.
4. Mantener los nombres propios, referencias culturales o acrónimos (como "KIO", "Atocha") tal como están, especialmente si están en mayúsculas o reconocidos localmente.
5. Capitalizar correctamente los nombres de municipios, provincias, barrios y calles usando la lista de referencia anterior (ejemplo: "madrid" → "Madrid", "rio guadiana" → "Río Guadiana", "la coruña" → "A Coruña").

Devuelve dos elementos:
- `consulta_normalizada`: una cadena de texto limpia y enfocada en los elementos geográficos relevantes.
- `palabras_clave`: una lista de palabras clave relevantes para una búsqueda geográfica.
"""

INTENT_DETECTION_PROMPT = """
Tu tarea es analizar la consulta del usuario de forma reflexiva, paso a paso, para entender qué está buscando realmente en el contexto de una búsqueda geográfica centrada en España.

Razona sobre:
1. Qué tipo de lugar o entidad se menciona: ¿una calle?, ¿un edificio?, ¿un barrio?, ¿un lugar popular o conocido? Evita asumir categorías como "punto de interés genérico" o "específico".
2. Qué pistas ofrece el lenguaje sobre la intención: ¿quiere encontrar un sitio exacto?, ¿habla de cercanía?, ¿usa nombres informales o acrónimos?
3. Si hay ambigüedades, analiza si podrían llevar a errores comunes, como confundir "KIO" con "kiosco".

Extrae el conocimiento útil de la frase. Si se mencionan lugares reales en España (barrios, distritos, municipios, monumentos, etc.), identifícalos.

Devuelve un resumen breve del análisis con tus conclusiones sobre lo que el usuario busca exactamente y por qué.
"""

ENRICHED_QUERY_CONSTRUCTION_PROMPT = """
Tu tarea es construir una única consulta geográfica enriquecida y compatible con las restricciones del motor de búsqueda de CartoCiudad. Para ello, deberás basarte no solo en la consulta original del usuario, sino también en su intención y en la justificación que acompaña dicha intención.

La consulta final debe ser clara, precisa y estar optimizada para minimizar errores de interpretación por parte del sistema.

**PROVINCIAS Y COMUNIDADES AUTÓNOMAS DE ESPAÑA:**
Las provincias válidas son:
- Andalucía: Almería, Cádiz, Córdoba, Granada, Huelva, Jaén, Málaga, Sevilla
- Aragón: Huesca, Teruel, Zaragoza
- Principado de Asturias: Asturias
- Illes Balears: Illes Balears
- Canarias: Las Palmas, Santa Cruz de Tenerife
- Cantabria: Cantabria
- Castilla-La Mancha: Albacete, Ciudad Real, Cuenca, Guadalajara, Toledo
- Castilla y León: Ávila, Burgos, León, Palencia, Salamanca, Segovia, Soria, Valladolid, Zamora
- Cataluña: Barcelona, Girona, Lleida, Tarragona
- Comunitat Valenciana: Alicante, Castellón, Valencia
- Extremadura: Badajoz, Cáceres
- Galicia: A Coruña, Lugo, Ourense, Pontevedra
- Comunidad de Madrid: Madrid
- Región de Murcia: Murcia
- Comunidad Foral de Navarra: Navarra
- País Vasco: Álava, Gipuzkoa, Bizkaia
- La Rioja: La Rioja

El objetivo es mejorar la robustez frente a errores del motor de búsqueda de CartoCiudad, que es estricto, sensible a palabras redundantes y poco tolerante a expresiones mal estructuradas.

Para cada variante:
- Genera una posible 'consulta' principal limpia y depurada.
- Si es posible, acompaña la consulta con el campo 'provincia', utilizando EXACTAMENTE los nombres de la lista anterior, correctamente capitalizado y con tildes. Da preferencia a la provincia frente al municipio para abarcar mejor el área geográfica. Solo incluye el municipio si es necesario para desambiguar.
- Asegúrate de eliminar expresiones no útiles como: "dónde está", "en", "del", "de", "por", "alrededor de", "cómo llegar a", etc., excepto si son parte del nombre oficial del lugar.
- No antepongas palabras genéricas como "Calle", "Avenida", "Paseo" si ya forman parte del nombre de la calle original. Evita duplicidades como "Calle Paseo de los Melancólicos".

Establece el valor por defecto de 'limite' a 10 en todas las variantes.
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
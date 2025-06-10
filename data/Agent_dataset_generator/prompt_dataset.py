QUERY_GENERATION_PROMPT = """
Eres un experto en generar consultas naturales de usuarios para direcciones geográficas.

Tu tarea es generar exactamente 5 variaciones de consultas que un usuario real podría hacer para buscar la dirección proporcionada.

Las variaciones deben incluir diferentes tipos de consultas:

1. **Directa**: La dirección tal como aparece o muy similar
2. **Natural**: Usando lenguaje más coloquial o natural
3. **Coloquial**: Con expresiones típicas del habla cotidiana
4. **Pregunta**: Formulada como pregunta explícita

Niveles de dificultad:
- **Fácil**: Consulta clara y directa, fácil de procesar
- **Medio**: Incluye alguna ambigüedad o términos coloquiales
- **Alto**: Muy coloquial, con posibles ambigüedades o información parcial

IMPORTANTE: 
- Las consultas deben ser realistas, como las que haría un usuario real
- Varía el nivel de formalidad y completitud
- Incluye diferentes formas de referirse al mismo lugar
- Considera que los usuarios pueden usar abreviaciones naturales
- Pueden incluir expresiones como "cerca de", "en", "por la zona de", etc.
- NO incluyas errores ortográficos en esta etapa

Devuelve exactamente 5 variaciones con sus respectivos tipos y niveles de dificultad.
"""

ERROR_INJECTION_PROMPT = """
Eres un experto en simular errores reales que cometen los usuarios al escribir consultas.

Tu tarea es tomar una consulta limpia y generar una versión con errores típicos de usuarios reales.

**IMPORTANTE: Debes seleccionar SOLO de estos 6 tipos de errores predefinidos:**

1. **mayusculas_inconsistentes**:
   - Todo en minúsculas: "Madrid" → "madrid"
   - Mayúsculas inconsistentes: "calle MAYOR", "CaLLe MayOR"
   - Todo en mayúsculas: "calle mayor" → "CALLE MAYOR"

2. **errores_ortograficos**:
   - Letras intercambiadas: "calle" → "clale"
   - Letras faltantes: "avenida" → "avendia", "donde" → "dond"
   - Letras extra: "plaza" → "plazza"

3. **espaciado_incorrecto**:
   - Espacios faltantes: "calle mayor" → "callemayor"
   - Espacios extra: "calle  mayor" (doble espacio)

4. **abreviaciones_incorrectas**:
   - "avenida" → "av", "avn", "avda", "avenda"
   - "calle" → "c/", "cl", "cal"
   - "plaza" → "pl", "pza"

5. **tildes_incorrectas**:
   - Tildes omitidas: "Móstoles" → "mostoles", "León" → "leon"
   - Tildes incorrectas: "donde" → "dónde"

6. **sustituciones_caracteres**:
   - "ñ" → "n": "España" → "Espana"
   - "ü" → "u": "Güemes" → "Guemes"
   - "ç" → "c": "Cáceres" → "Caceres"

**REGLAS:**
- Introduce entre 1-3 errores por consulta
- Los errores deben ser realistas y naturales
- No hagas la consulta incomprensible
- Mantén la intención original clara
- Debes devolver una lista con los tipos exactos aplicados (usando los nombres arriba especificados)

**Formato de respuesta:**
- query_with_errors: La consulta con errores aplicados
- error_types: Lista con los tipos exactos aplicados (ej: ["mayusculas_inconsistentes", "tildes_incorrectas"])
""" 
import sys
import os
import json
import random
import time

# Permitir imports relativos desde el proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.common.tools import search_cartociudad_tool
from data.constantes import provincias_de_espana_por_comunidad

# Parámetros
TIPOS_EXCLUIR = ["expendeduria", "punto_recarga_electrica"]
TERMINOS_BUSQUEDA = ["calle", "paseo", "avenida", "plaza"]
MAX_RESULTADOS_POR_TERMINO = 20
OUTPUT_FILE = "data/datasets/dataset_direcciones_init.json"


def obtener_lista_provincias():
    provincias = []
    for lista in provincias_de_espana_por_comunidad.values():
        provincias.extend(lista)
    return sorted(set(provincias))


def buscar_direcciones_en_provincia(provincia):
    resultados = []
    for termino in TERMINOS_BUSQUEDA:
        consulta = termino
        try:
            candidatos = search_cartociudad_tool(
                consulta=consulta,
                limite=MAX_RESULTADOS_POR_TERMINO,
                excluir_tipos=TIPOS_EXCLUIR,
                provincia=provincia
            )
            print(f"[DEBUG] {provincia} - {termino}: {len(candidatos)} resultados brutos")
            if candidatos:
                print(f"[DEBUG] Ejemplo: {candidatos[0]}")
        except Exception as e:
            print(f"[ERROR] Provincia: {provincia}, término: {termino} -> {e}")
            candidatos = []
        # Añadir tipo_consulta a cada resultado
        for cand in candidatos:
            cand["provincia"] = provincia
            cand["tipo_consulta"] = termino
        resultados.extend(candidatos)
        time.sleep(random.uniform(1, 2))
    print(f"[DEBUG] Total resultados antes de deduplicar para {provincia}: {len(resultados)}")
    return resultados


def deduplicar_por_address_id(candidates):
    unique = {}
    for cand in candidates:
        key = cand.get('address') or cand.get('id')
        if key and key not in unique:
            unique[key] = cand
    return list(unique.values())


def main():
    provincias = obtener_lista_provincias()
    dataset = []
    resumen = {}
    if os.path.exists(OUTPUT_FILE):
        print(f"\n[AVISO] El fichero '{OUTPUT_FILE}' ya existe. No se sobrescribirá. Elimina o renombra el fichero para generar uno nuevo.")
        return
    # Crear carpeta 'datasets' si no existe
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    for provincia in provincias:
        print(f"Procesando provincia: {provincia} ...")
        resultados = buscar_direcciones_en_provincia(provincia)
        print(f"[DEBUG] {provincia}: {len(resultados)} resultados antes de deduplicar")
        resultados_dedup = deduplicar_por_address_id(resultados)
        print(f"[DEBUG] {provincia}: {len(resultados_dedup)} resultados después de deduplicar")
        dataset.extend(resultados_dedup)
        resumen[provincia] = len(resultados_dedup)
        print(f"  -> {len(resultados_dedup)} direcciones únicas")
    # Guardar dataset
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    print("\nResumen de direcciones recogidas por provincia:")
    for provincia, n in resumen.items():
        print(f"{provincia}: {n}")
    print(f"\nTotal de direcciones en el dataset: {len(dataset)}")
    print(f"Dataset guardado en: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
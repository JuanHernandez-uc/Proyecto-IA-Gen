#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Consulta Wikidata para obtener los tipos ontológicos (P31 / P279)
de las entidades listadas en un archivo CSV por país.
"""

import argparse
import os
import time
import pandas as pd
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "kg-pipeline/1.0 (dlarraguibel@uc.cl; academic use)"
PREFIXES = """
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

def _escape_literal(s: str) -> str:
    return s.replace('\\', '\\\\').replace('"', '\\"')

def _sparql():
    s = SPARQLWrapper(SPARQL_ENDPOINT, agent=USER_AGENT)
    s.setReturnFormat(JSON)
    s.setTimeout(60)
    return s

def qid_from_label(label, max_retries=3, base_sleep=0.1):
    """Busca el QID de una entidad por su etiqueta (es/en)."""
    
    last_err = None
    query = f"""
    {PREFIXES}
    SELECT ?item ?itemLabel WHERE {{
      # 1) match exacto en español
      {{ ?item rdfs:label "{_escape_literal(label)}"@es. }}
      UNION
      # 2) si no hay en es, intenta en inglés
      {{ ?item rdfs:label "{_escape_literal(label)}"@en. }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en". }}
    }}
    LIMIT 5
    """

    for attempt in range(max_retries):
        try:
            sparql = _sparql()
            sparql.setQuery(query)
            results = sparql.query().convert()
            time.sleep(0.1)
            
            bindings = results.get("results", {}).get("bindings", [])
            if not bindings:
                return None
            qid = bindings[0]["item"]["value"].rsplit("/", 1)[-1]
            qlabel = bindings[0].get("itemLabel", {}).get("value", qid)
            return qid, qlabel

        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** attempt))

    print(f"[qid_from_label] Falló con '{label}': {last_err}", flush=True)
    return None

def query_entity_type(qid, max_retries=3, base_sleep=0.1):
    """Obtiene P31 (instancia de) y P279 (subclase de) para un QID."""
    last_err = None
    query = f"""
    {PREFIXES}
    SELECT ?prop ?type ?typeLabel WHERE {{
      VALUES ?item {{ wd:{qid} }}
      {{
        ?item wdt:P31 ?type .
        BIND("P31" AS ?prop)
      }} UNION {{
        ?item wdt:P279 ?type .
        BIND("P279" AS ?prop)
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es,en". }}
    }}
    """
    for attempt in range(max_retries):
        try:
            sparql = _sparql()
            sparql.setQuery(query)
            results = sparql.query().convert()
            # Pausa tras cada consulta
            time.sleep(0.1)

            out = {"instance_of": [], "subclass_of": []}
            for b in results.get("results", {}).get("bindings", []):
                prop = b["prop"]["value"]
                t_qid = b["type"]["value"].rsplit("/", 1)[-1]
                t_label = b.get("typeLabel", {}).get("value", t_qid)
                if prop == "P31":
                    out["instance_of"].append((t_qid, t_label))
                elif prop == "P279":
                    out["subclass_of"].append((t_qid, t_label))
            return out

        except Exception as e:
            last_err = e
            time.sleep(base_sleep * (2 ** attempt))
    print(f"[query_entity_type] Falló con {qid}: {last_err}", flush=True)
    return {"instance_of": [], "subclass_of": []}


def main(country_name):
    """Procesa todas las entidades del CSV correspondiente al país dado."""
    input_csv = f"Datasets/{country_name}.csv"
    output_dir = "Datasets/entity_types"
    output_file = os.path.join(output_dir, f"{country_name}.tsv")

    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_csv)
    entidades = df["entidad"].dropna().unique()

    entity_types = {}
    for entity in tqdm(entidades, desc=f"Consultando {country_name}"):
        qid_info = qid_from_label(entity)
        if qid_info:
            qid, _ = qid_info
            types = query_entity_type(qid)
        else:
            types = {"instance_of": [], "subclass_of": []}
        entity_types[entity] = types
        time.sleep(0.1) 

    records = []
    for ent, types in entity_types.items():
        inst = ", ".join([lbl for _, lbl in types["instance_of"]]) or None
        subc = ", ".join([lbl for _, lbl in types["subclass_of"]]) or None
        records.append({"entidad": ent, "instancia_de": inst, "subclase_de": subc})

    out_df = pd.DataFrame(records)
    out_df.to_csv(output_file, sep="\t", index=False, encoding="utf-8")
    print(f"\nResultados guardados en: {output_file}")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Consulta tipos ontológicos en Wikidata por país.")
#     parser.add_argument("country_name", help="Nombre del país (coincide con el archivo Datasets/<country>.csv)")
#     args = parser.parse_args()
#     country_name = args.country_name.lower()   
#     main(country_name)

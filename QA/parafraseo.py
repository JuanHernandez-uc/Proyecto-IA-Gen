from SPARQLWrapper import SPARQLWrapper, JSON
import time
import pandas as pd

ENDPOINT = "https://query.wikidata.org/sparql"
DEFAULT_LANG = "es"


def make_sparql():
    s = SPARQLWrapper(ENDPOINT)
    s.setReturnFormat(JSON)
    return s


def run_sparql(query, max_retries=3, backoff=1.5):
    """Ejecuta una consulta SPARQL con reintentos simples."""
    sparql = make_sparql()
    sparql.setQuery(query)
    last_err = None
    for attempt in range(max_retries):
        try:
            return sparql.query().convert()
        except Exception as e:
            last_err = e
            time.sleep(backoff ** attempt)
    raise last_err


def get_qid_via_sparql(term, lang=DEFAULT_LANG):
    """Obtiene el QID de un término usando el servicio EntitySearch dentro del endpoint SPARQL."""
    query = f"""
    SELECT ?item ?rank
    WHERE {{
      SERVICE wikibase:mwapi {{
        bd:serviceParam wikibase:endpoint "www.wikidata.org";
                         wikibase:api "EntitySearch";
                         mwapi:search "{term}";
                         mwapi:language "{lang}".
        ?item wikibase:apiOutputItem mwapi:item .
        ?rank wikibase:apiOrdinal true .
      }}
    }}
    ORDER BY ?rank
    LIMIT 1
    """
    data = run_sparql(query)
    bindings = data["results"]["bindings"]
    if not bindings:
        return None
    uri = bindings[0]["item"]["value"]
    return uri.rsplit("/", 1)[-1]  # devuelve el QID, p.ej. 'Q2887'


def get_aliases_by_qid(qid, lang=DEFAULT_LANG):
    """Devuelve la lista de alias de un QID en el idioma dado."""
    query = f"""
    SELECT ?alias
    WHERE {{
      wd:{qid} skos:altLabel ?alias .
      FILTER(LANG(?alias) = "{lang}")
    }}
    """
    data = run_sparql(query)
    return [r["alias"]["value"] for r in data["results"]["bindings"]]


def get_aliases_from_name(term, lang=DEFAULT_LANG):
    """
    Dado un string (nombre de una entidad), busca su QID y retorna una lista con todos sus aliases.
    Si no encuentra nada, devuelve [].
    """
    qid = get_qid_via_sparql(term, lang)
    if not qid:
        print(f"No se encontró QID para '{term}'.")
        return []
    aliases = get_aliases_by_qid(qid, lang)
    return aliases


if __name__ == "__main__":
    import os
    import ast
    import json
    import time
    import pandas as pd
    from tqdm import tqdm
    # ===================== Config =====================
    INPUT_PATH = "QA/QA_dataset.csv"
    OUTPUT_PATH = "QA/QA_dataset_aliases_parcial.csv"
    CACHE_PATH = "QA/aliases_cache.json"
    LANG = "es"

    # Guardado periódico del caché (cada N términos únicos resueltos)
    CACHE_CHECKPOINT_EVERY = 200

    tqdm.pandas(desc="Obteniendo aliases de Wikidata")


    # ===================== Utilidades =====================
    def parse_list_like(x):
        """Convierte strings tipo '["...","..."]' a lista real; deja intacto lo demás."""
        if isinstance(x, str) and x.startswith("["):
            try:
                return ast.literal_eval(x)
            except Exception:
                return x  # si no parsea, lo dejamos como estaba
        return x


    def safe_get_aliases(term: str, lang=LANG):
        """Wrapper con manejo de errores para pedir aliases de un término (string)."""
        try:
            return get_aliases_from_name(term, lang=lang)
        except Exception as e:
            print(f"Error resolviendo '{term}': {e}")
            return []


    def atomic_json_dump(obj, path):
        """Guarda JSON de forma atómica (tmp + replace)."""
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False)
        os.replace(tmp, path)


    def atomic_csv_dump(df, path):
        """Guarda CSV de forma atómica (tmp + replace)."""
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)


    # ===================== Carga dataset =====================
    # Si existe OUTPUT, partimos desde ahí (mantiene mismo orden/columnas).
    if os.path.exists(OUTPUT_PATH):
        dataset = pd.read_csv(OUTPUT_PATH)
    else:
        dataset = pd.read_csv(INPUT_PATH)

    # Asegurar tipos correctos
    if "respuestas" not in dataset.columns:
        raise ValueError("No se encontró la columna 'respuestas' en el dataset.")

    dataset["respuestas"] = dataset["respuestas"].apply(parse_list_like)

    # Crear col de salida si no existe (la vamos a rellenar completa desde caché)
    if "respuestas_aliases" not in dataset.columns:
        dataset["respuestas_aliases"] = None


    # ===================== Carga caché =====================
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            aliases_cache = json.load(f)
    else:
        aliases_cache = {}

    # Normalizar claves (espacios de más)
    aliases_cache = { (k.strip() if isinstance(k, str) else k): v for k, v in aliases_cache.items() }


    # ===================== Construir términos únicos =====================
    unique_terms = set()
    for val in dataset["respuestas"]:
        if isinstance(val, list):
            for item in val:
                if isinstance(item, str) and item.strip():
                    unique_terms.add(item.strip())
        elif isinstance(val, str) and val.strip():
            unique_terms.add(val.strip())

    print(f"Términos únicos detectados: {len(unique_terms):,}")

    # Faltantes en caché
    missing_terms = [t for t in unique_terms if t not in aliases_cache]
    print(f"Términos faltantes por resolver: {len(missing_terms):,}")


    # ===================== Resolver faltantes (una sola vez) =====================
    if missing_terms:
        for i, term in enumerate(tqdm(missing_terms, desc="Resolviendo términos únicos")):
            aliases_cache[term] = safe_get_aliases(term, lang=LANG)

            # Guardado periódico del caché
            if (i + 1) % CACHE_CHECKPOINT_EVERY == 0 or (i + 1) == len(missing_terms):
                atomic_json_dump(aliases_cache, CACHE_PATH)
                # Pequeña pausa para cortesía / evitar saturar I/O o endpoint
                time.sleep(0.2)
    else:
        # Asegurar que el caché esté persistido si no existía
        if not os.path.exists(CACHE_PATH):
            atomic_json_dump(aliases_cache, CACHE_PATH)


    # ===================== Rellenar columna desde caché (rápido) =====================
    def fill_row(value):
        """
        - Si la fila es list[str], retorna list[list[str]] con aliases por cada string.
        - Si la fila es str, retorna list[str] de aliases de ese string.
        - Si es NaN/u otro tipo, retorna [].
        """
        if isinstance(value, list):
            out = []
            for item in value:
                if isinstance(item, str):
                    out.append(aliases_cache.get(item.strip(), []))
                else:
                    out.append([])
            return out
        elif isinstance(value, str):
            return aliases_cache.get(value.strip(), [])
        else:
            return []

    dataset["respuestas_aliases"] = dataset["respuestas"].progress_apply(fill_row)

    # ===================== Guardar resultado =====================
    atomic_csv_dump(dataset, OUTPUT_PATH)
    print(f"Listo. Guardado en: {OUTPUT_PATH}")
    print(f"Caché persistido en: {CACHE_PATH}")
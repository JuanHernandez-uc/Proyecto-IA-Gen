import pandas as pd
from pathlib import Path
import re
import unicodedata
import pandas as pd

STOP_WORDS = {
    # Artículos y determinantes
    "el", "la", "los", "las", "lo", "un", "una", "unos", "unas", "al", "del",

    # Preposiciones y conjunciones
    "a", "ante", "bajo", "con", "contra", "de", "desde", "en", "entre", "hacia",
    "hasta", "para", "por", "segun", "sin", "sobre", "tras", "y", "o", "u", "e", "ni", "pero",

    # Pronombres y posesivos
    "mi", "mis", "tu", "tus", "su", "sus", "nuestro", "nuestra", "nuestros", "nuestras",
    "vuestro", "vuestra", "vuestros", "vuestras", "se", "que", "quien", "cual", "cuales",
    "de", "del", "la", "las", "el", "los", "en", "al", "por", "para", "su", "sus", "es", "como",
}

relaciones_a_eliminar = [
    "fabricante",
    "ubicado en la entidad territorial administrativa actual",
    "cargo ocupado por el dirigente",
    "prefijo honorifico",
    "punto mas alto",
    "signatario",
    "clasificacion cnc (francia)",
    "examen realizado",
    "calificacion medieradet",
    "lugar de ambientacion",
    "director ejecutivo",
    "clima",
    "del universo narrativo",
    "presente en la obra",
    "orden religiosa",
    "situado en la calle",
    "nucleo de poblacion",
    "instrumentacion",
    "uso",
    "caracteristica de",
    "editor",
    "inspirado por",
    "mantenido por",
    "donado por",
    "sistema de escritura",
    "organo legislativo",
    "alimentacion",
    "desembocadura del curso de agua",
    "afluente",
    "cuenca hidrografica",
    "punto mas bajo",
    "lado de conduccion",
    "categoria para las peliculas rodadas aqui",
    "situado en la entidad territorial estadistica",
    "lista de monumentos",
    "estatuto de observador oficial en la organizacion",
    "tipo de enchufe electrico",
    "dia del año de una ocurrencia periodica",
    "diocesis",
    "album de banda sonora",
    "emisora original",
    "personajes",
    "moneda",
    "tiene obras en la coleccion",
    "participo en el conflicto",
    "contiene la entidad territorial estadistica",
    "estilo arquitectonico",
    "categoria para los mapas de este elemento"
]

def _clean_text(text):
    """Normaliza texto: minúsculas, sin tildes ni puntuación, y sin stopwords."""
    if not isinstance(text, str):
        return ""
    text = remove_accents(text).lower()
    # Remueve puntuación y símbolos simples
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)

def _entity_in_objects(entidad, objetos):
    """True si la entidad (limpia) es substring de alguno de los objetos (limpios)."""
    entidad_clean = _clean_text(entidad)
    if not entidad_clean:
        return False

    if isinstance(objetos, list):
        return any(
            isinstance(o, str) and entidad_clean in _clean_text(o)
            for o in objetos
        )
    elif isinstance(objetos, str):
        return entidad_clean in _clean_text(objetos)
    return False

def remove_accents(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
        
def normalize_columns(df, columns):
    """
    Normaliza texto en columnas seleccionadas de un DataFrame:
    convierte a minúsculas y elimina tildes/acentos.
    Compatible con celdas que contengan strings, listas o listas de listas.
    """
    def _normalize_cell(value):
        if isinstance(value, str):
            return remove_accents(value)
        elif isinstance(value, list):
            return [_normalize_cell(v) for v in value]  # recursión
        else:
            return value

    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = df[col].apply(_normalize_cell)
    return df

def filter_data(df):
    # filtros
    
    ## por entidad
    df = df[~df["entidad"].str.match(r"^Q\d+$", na=False)] # Entidades que quedaron como QID
    df = df[~df["entidad"].str.startswith("Anexo:", na=False)]

    ## por relación
    df = df[~df["relacion"].str.match(r"^P\d+$", na=False)] # Relaciones que quedaron como QID
    df = df[~df["relacion"].str.lower().str.startswith("subdividido en", na=False)] # clase inútil
    df = df[~df["relacion"].str.lower().str.startswith("opuesto", na=False)]
    df = df[~df["relacion"].str.lower().str.startswith(remove_accents("categoría"), na=False)]
    df = df[~df["relacion"].str.lower().str.startswith(remove_accents("rango inmediatamente"), na=False)]
    df = df[~df["relacion"].str.lower().str.startswith(remove_accents("organización dirigida desde"), na=False)]
    df = df[~df["relacion"].str.lower().str.startswith(remove_accents("edición"), na=False)]
    df = df[~df["relacion"].str.lower().str.startswith(remove_accents("traduccion"), na=False)]
    df = df[~df["relacion"].str.lower().str.startswith(remove_accents("lista de"), na=False)]

    df = df[df["relacion"].str.lower() != remove_accents("se dice que es lo mismo que")]
    df = df[df["relacion"].str.lower() != remove_accents("descrito en la fuente")]
    df = df[df["relacion"].str.lower() != remove_accents("se encuentra en el huso horario")]
    df = df[df["relacion"].str.lower() != remove_accents("lista de interés para el proyecto Wikimedia")]
    df = df[df["relacion"].str.lower() != remove_accents("forma jurídica")]
    df = df[df["relacion"].str.lower() != remove_accents("calendario académico")]
    df = df[df["relacion"].str.lower() != remove_accents("forma artística")]
    df = df[df["relacion"].str.lower() != remove_accents("sucedido por")]
    df = df[df["relacion"].str.lower() != remove_accents("operador")]
    df = df[df["relacion"].str.lower() != remove_accents("diferente de")]
    df = df[df["relacion"].str.lower() != remove_accents("vestimenta")]
    df = df[df["relacion"].str.lower() != remove_accents("caracterizado por")]
    df = df[df["relacion"].str.lower() != remove_accents("precedido por")]
    df = df[df["relacion"].str.lower() != remove_accents("faceta de")]
    df = df[df["relacion"].str.lower() != remove_accents("lista del elemento")]
    df = df[df["relacion"].str.lower() != remove_accents("cargo ocupado por el jefe de gobierno")]
    df = df[df["relacion"].str.lower() != remove_accents("texto regulador principal")]
    df = df[df["relacion"].str.lower() != remove_accents("secretario general")]
    df = df[df["relacion"].str.lower() != remove_accents("industria")]
    df = df[df["relacion"].str.lower() != remove_accents("escudo de armas")]
    df = df[df["relacion"].str.lower() != remove_accents("propietario de")]
    df = df[df["relacion"].str.lower() != remove_accents("propiedad de")]
    df = df[df["relacion"].str.lower() != remove_accents("título nobiliario")]
    df = df[df["relacion"].str.lower() != remove_accents("confiere")]
    df = df[df["relacion"].str.lower() != remove_accents("reemplaza a")]
    df = df[df["relacion"].str.lower() != remove_accents("afiliacion")]
    df = df[df["relacion"].str.lower() != remove_accents("serie")]
    df = df[df["relacion"].str.lower() != remove_accents("bandera")]
    df = df[df["relacion"].str.lower() != remove_accents("geografia")]
    df = df[df["relacion"].str.lower() != remove_accents("estatus patrimonial")]
    df = df[df["relacion"].str.lower() != remove_accents("historia")]
    df = df[df["relacion"].str.lower() != remove_accents("cultura")]
    df = df[df["relacion"].str.lower() != remove_accents("telefono de emergencia")]
    df = df[df["relacion"].str.lower() != remove_accents("cantidad fisica medida")]
    df = df[df["relacion"].str.lower() != remove_accents("organizador")]
    df = df[df["relacion"].str.lower() != remove_accents("movimiento")]
    df = df[df["relacion"].str.lower() != remove_accents("coleccion")]
    df = df[df["relacion"].str.lower() != remove_accents("sexo o genero")]
    df = df[df["relacion"].str.lower() != remove_accents("pais de nacionalidad")]
    df = df[df["relacion"].str.lower() != remove_accents("nombre de pila")]
    df = df[df["relacion"].str.lower() != remove_accents("parcialmente coincidente con")]
    df = df[df["relacion"].str.lower() != remove_accents("situado en la entidad geografica")]
    df = df[df["relacion"].str.lower() != remove_accents("formato de periodico")]
    df = df[df["relacion"].str.lower() != remove_accents("manifestacion de")]
    df = df[df["relacion"].str.lower() != remove_accents("estadio")]
    df = df[df["relacion"].str.lower() != remove_accents("liga")]
    df = df[df["relacion"].str.lower() != remove_accents("arquitecto")]
    df = df[df["relacion"].str.lower() != remove_accents("condado historico")]
    df = df[df["relacion"].str.lower() != remove_accents("color oficial")]
    df = df[df["relacion"].str.lower() != remove_accents("representa a")]
    df = df[df["relacion"].str.lower() != remove_accents("basado en")]
    df = df[df["relacion"].str.lower() != remove_accents("estatus de los derechos de autor")]
    df = df[df["relacion"].str.lower() != remove_accents("instancia tiene partes de la clase")]
    df = df[df["relacion"].str.lower() != remove_accents("presentador")]
    df = df[df["relacion"].str.lower() != remove_accents("suplente del cargo")]
    df = df[df["relacion"].str.lower() != remove_accents("utiliza")]
    df = df[df["relacion"].str.lower() != remove_accents("descripcion del sello")]
    df = df[df["relacion"].str.lower() != remove_accents("reemplazado por")]
    df = df[df["relacion"].str.lower() != remove_accents("portal principal del tema")]
    df = df[df["relacion"].str.lower() != remove_accents("banco central")]
    df = df[df["relacion"].str.lower() != remove_accents("empresa productora")]
    df = df[df["relacion"].str.lower() != remove_accents("idioma de la película")]
    df = df[df["relacion"].str.lower() != remove_accents("programa de televisión")]
    
    RELACIONES_A_ELIMINAR_NORM = {
        remove_accents(s).strip().lower() for s in relaciones_a_eliminar
    }
    df = df[~df["relacion"].apply(
        lambda x: remove_accents(str(x)).strip().lower() in RELACIONES_A_ELIMINAR_NORM
        )]

    # por objetos (son listas de strings)
    def _has_http(value):
        if isinstance(value, list):
            return any(isinstance(v, str) and v.startswith("http") for v in value)
        return isinstance(value, str) and value.startswith("http")

    def _starts_with_categoria(value):
        if isinstance(value, list):
            return any(isinstance(v, str) and remove_accents(v).startswith("categoria") for v in value)
        return isinstance(value, str) and remove_accents(value).startswith("categoria")


    def _contains_designado_por(value):
        if isinstance(value, list):
            return any(isinstance(v, str) and "designado por" in remove_accents(v) for v in value)
        return isinstance(value, str) and "designado por" in remove_accents(value)


    df = df[~df["objetos"].apply(lambda x: any(isinstance(v, str) and remove_accents(v) == "titulo nobiliario" for v in x) if isinstance(x, list) else isinstance(x, str) and remove_accents(x) == "titulo nobiliario")]
    df = df[~df["objetos"].apply(lambda x: any(isinstance(v, str) and remove_accents(v) == "creador" for v in x) if isinstance(x, list) else isinstance(x, str) and remove_accents(x) == "creador")]

    mask_no_http = ~df["objetos"].apply(_has_http)
    mask_no_categoria = ~df["objetos"].apply(_starts_with_categoria)
    mask_no_designado = ~df["objetos"].apply(_contains_designado_por)
    df = df[mask_no_http & mask_no_categoria & mask_no_designado]

    # drop duplicates que funciona con lista objetos
    df = df.assign(objetos=df["objetos"].apply(lambda x: tuple(x) if isinstance(x, list) else x)) \
       .drop_duplicates(subset=["entidad", "relacion", "objetos"], keep="first")

    df = df[~df.apply(lambda row: _entity_in_objects(row["entidad"], row["objetos"]), axis=1)]

    return df


folder = Path('QA')
df = pd.read_csv(folder / 'QA_dataset_aliases_filtrados.csv')

print(f"\nInitial dataset size: {len(df)} rows")
df_normalized = normalize_columns(df, ['entidad', 'relacion', 'objetos', 'respuestas', 'respuestas_aliases'])
df_filtered = filter_data(df_normalized)
(folder / 'golden_QA_dataset.csv').write_text(df_filtered.to_csv(index=False))
print(f'Filtered dataset size: {len(df_filtered)} rows')
print(f'porcentaje de filas eliminadas: {100 * (1 - len(df_filtered) / len(df)):.2f}%\n')
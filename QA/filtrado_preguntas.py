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


relaciones_a_mantener = [
    "deporte",
    "distrito escolar",
    "situado cerca del cuerpo de agua",
    "sello discografico",
    "comparte fronteras con",
    "genero",
    "himno",
    "idioma oficial",
    "capital",
    "ciudadania",
    "fecha de nacimiento",
    "forma de gobierno",
    "ocupacion",
    "organizacion matriz",
    "religion",
    "residencia oficial",
    "fecha de defuncion",
    "capital de",
    "miembro de",
    "educado en",
    "lugar de sepultura",
    "lugar de publicacion",
    "lugar de nacimiento",
    "ubicacion",
    "ubicacion de la sede",
    "lugar de formacion",
    "lugar de trabajo",
    "pais de origen",
    "nombrado en referencia a",
    "continente",
    "situado en la entidad territorial administrativa",
    "idioma de la obra o del nombre",
    "idioma de la pelicula o programa de television",
    "pais",
    "musica de",
    "conferido por",
    "creador",
    "empleador",
    "gerente/director",
    "ilustrador",
    "jefe de gobierno",
    "rector",
    "santo patron",
    "actor de voz",
    "autor",
    "jefe de estado",
    "presidente",
    "productor",
    "guionista",
    "letra de",
    "director",
    "financiador",
    "fundador",
    "interprete",
    "cargo",
    "color",
    "editorial",
    "evento significativo",
    "idioma usado",
    "premio recibido", 
]

def remove_accents(text):
    if not isinstance(text, str):
        return text
    text = text.lower()
    return ''.join(
        c for c in unicodedata.normalize('NFKD', text)
        if not unicodedata.combining(c)
    )
    
RELACIONES_A_MANTENER_NORM = {
    remove_accents(s).strip().lower() for s in relaciones_a_mantener
}

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
    
def _clean_text(text):
    """Normaliza texto: minúsculas, sin tildes ni puntuación, y sin stopwords."""
    if not isinstance(text, str):
        return ""
    text = remove_accents(text).lower()
    # Remueve puntuación y símbolos simples
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(tokens)


def entidad_contenida_en_respuesta(entidad, respuestas):
    """True si la entidad (limpia) es substring de alguna de las respuestas (limpias) o vice versa."""
    entidad_clean = _clean_text(entidad)
    if not entidad_clean:
        return False

    if isinstance(respuestas, list):
        return any(
            isinstance(r, str) and (entidad_clean in _clean_text(r) or _clean_text(r) in entidad_clean)
            for r in respuestas
        )
    elif isinstance(respuestas, str):
        return entidad_clean in _clean_text(respuestas) or _clean_text(respuestas) in entidad_clean
    return False



def filter_data(df):

    df = df[~df["entidad"].str.match(r"^Q\d+$", na=False)] # Entidades que quedaron como QID
    df = df[~df["entidad"].str.startswith("Anexo:", na=False)] # entidades que empeizan con "Anexo:"
    
    # eliminamos objetos que contienen http, empiezan con cateogoria o designado
    mask_no_http = ~df["objetos"].apply(lambda v: any(isinstance(x, str) and x.startswith("http") for x in (v if isinstance(v, list) else [v])))
    mask_no_categoria = ~df["objetos"].apply(lambda v: any(isinstance(x, str) and remove_accents(x).startswith("categoria") for x in (v if isinstance(v, list) else [v])))
    mask_no_designado = ~df["objetos"].apply(lambda v: any(isinstance(x, str) and remove_accents(x).startswith("designado") for x in (v if isinstance(v, list) else [v])))
    df = df[mask_no_http & mask_no_categoria & mask_no_designado]

    # eliminamos relaciones indeseadas
    df = df[df["relacion"].apply(
        lambda x: remove_accents(str(x)).strip().lower() in RELACIONES_A_MANTENER_NORM
    )]

    df = df[~df["relacion"].apply(lambda x: isinstance(x, str) and remove_accents(x).strip().lower().startswith("categoria"))]

    # eliminamos respuestas contenidas en la pregunta y vice versa
    df = df[~df.apply(lambda row: entidad_contenida_en_respuesta(row["entidad"], row["respuestas"]), axis=1)]

    # drop duplicates que funciona con lista objetos
    df = df.assign(objetos=df["objetos"].apply(lambda x: tuple(x) if isinstance(x, list) else x)) \
       .drop_duplicates(subset=["entidad", "relacion", "objetos"], keep="first")
       
    return df

folder = Path('QA')
df = pd.read_csv(folder / 'QA_dataset_aliases_filtrados.csv')

print(f"\nInitial dataset size: {len(df)} rows")
df_normalized = normalize_columns(df, ['relacion', 'objetos', 'respuestas', 'respuestas_aliases'])
df_filtered = filter_data(df_normalized)
(folder / 'golden_QA_dataset.csv').write_text(df_filtered.to_csv(index=False))
print(f'Filtered dataset size: {len(df_filtered)} rows')
print(f'porcentaje de filas eliminadas: {100 * (1 - len(df_filtered) / len(df)):.2f}%\n')
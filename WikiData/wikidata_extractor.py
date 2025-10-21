from SPARQLWrapper import SPARQLWrapper, JSON
import pandas as pd

# Conexión con WikiData
sparql = SPARQLWrapper("https://query.wikidata.org/sparql")

# Query
sparql.setQuery("""
SELECT  
    ?persona ?personaLabel 
    ?lugarNacimiento ?lugarNacimientoLabel 
    ?ocupacion ?ocupacionLabel 
    ?premio ?premioLabel 
    ?fechaNac ?fechaDef
    ?ciudadania ?ciudadaniaLabel
    ?educadoEn ?educadoEnLabel
    ?empleador ?empleadorLabel
    ?obraNotable ?obraNotableLabel
    ?miembroDe ?miembroDeLabel
    ?cargo ?cargoLabel
WHERE {
  {
    SELECT DISTINCT ?persona WHERE {
      ?persona wdt:P31 wd:Q5;   # Instancia de ser humano
               wdt:P27 wd:Q96.  # Con ciudadanía mexicana
    }
    LIMIT 10
  }

  # --- Relaciones opcionales de cada persona ---
  OPTIONAL { ?persona wdt:P19  ?lugarNacimiento. }   # Lugar de nacimiento
  OPTIONAL { ?persona wdt:P106 ?ocupacion. }          # Ocupación
  OPTIONAL { ?persona wdt:P166 ?premio. }             # Premio
  OPTIONAL { ?persona wdt:P569 ?fechaNac. }           # Fecha de nacimiento
  OPTIONAL { ?persona wdt:P570 ?fechaDef. }           # Fecha de defunción
  OPTIONAL { ?persona wdt:P27  ?ciudadania. }         # Ciudadanía
  OPTIONAL { ?persona wdt:P69  ?educadoEn. }          # Educado en
  OPTIONAL { ?persona wdt:P108 ?empleador. }          # Empleador
  OPTIONAL { ?persona wdt:P800 ?obraNotable. }        # Obra notable
  OPTIONAL { ?persona wdt:P463 ?miembroDe. }          # Miembro de
  OPTIONAL { ?persona wdt:P39  ?cargo. }              # Cargo

  SERVICE wikibase:label { bd:serviceParam wikibase:language "es,en". }
}
""")

# Formato JSON para hacer la request
sparql.setReturnFormat(JSON)

print("Realizando request")

# Ejecuta la consulta
results = sparql.query().convert()

print("Request realizada")

# Convertirlo en DataFrame
df = pd.json_normalize(results["results"]["bindings"])

# Columnas consultadas
names_mapping = {
    "lugarNacimientoLabel.value": "Lugar de nacimiento",
    "ocupacionLabel.value": "Ocupación",
    "ciudadaniaLabel.value": "Ciudadanía",
    "educadoEnLabel.value": "Educado en",
    "empleadorLabel.value": "Empleador",
    "obraNotableLabel.value": "Obra notable",
    "miembroDeLabel.value": "Miembro de",
    "cargoLabel.value": "Cargo",
}
optional_names_mapping = {
    "premioLabel.value": "Premio recibido"
}
date_mapping = {  # literales (sin label)
    "fechaNac.value": "Fecha de nacimiento",
    "fechaDef.value": "Fecha de defunción",
}

# Crear tripletas
triplets = []

# Se verifica que exista (la relación podría no existir). En el caso de las fechas no hay label porque son literales
for _, row in df.iterrows():
    entidad = row['personaLabel.value']

    for label, name in names_mapping.items():
        if label in row:
            triplets.append([entidad, name, row[label], row[label.replace("Label.value", ".value")]])
    
    for label, name in optional_names_mapping.items():
        if label in row and str(row.get(label)) != "nan":
            triplets.append([entidad, name, row[label], row[label.replace("Label.value", ".value")]])

    for date_key, name in date_mapping.items():
        value = row.get(date_key)
        if isinstance(value, str):
            obj = value.split("T")[0]
            triplets.append([entidad, name, obj, None])

# Lo convertimos en dataframe y sacamos duplicados
df_triplets = pd.DataFrame(triplets, columns=["entidad", "relacion", "objeto", "uri_object"]).drop_duplicates()

# Función para explorar en profundidad. Con profundidad 1 solo se hace la consulta anterior, con profundidad > 2 empieza a explorar.
# Solo explora considerando dos objetos por cada entidad obtenida, para que no explote jjsdjsdj
def expand_level(qids, depth=1, extension_limit = 2):
    all_triplets = []

    frontier = qids

    for i in range(2, depth + 1):
        print("Ejecutando profundidad:", i)
        # Caso en que no tengamos más entidades por explorar
        if not frontier:
            break
        
        # Dataframe en el que guardaremos la información de cada objeto
        df2_parts = []

        # Consultamos máximo 5 relaciones posibles por objeto
        for s_uri in frontier:
            query = f"""
            PREFIX wdt: <http://www.wikidata.org/prop/direct/>
            PREFIX wd: <http://www.wikidata.org/entity/>
            PREFIX wikibase: <http://wikiba.se/ontology#>

            SELECT ?s ?sLabel ?p ?prop ?propLabel ?o ?oLabel WHERE {{
            BIND(<{s_uri}> AS ?s)

            ?s ?p ?o .
            FILTER(STRSTARTS(STR(?p), STR(wdt:)))
            FILTER(STRSTARTS(STR(?o), STR(wd:Q)))

            # mapear wdt:Pxx -> wd:Pxx (entidad propiedad) para obtener label
            ?prop wikibase:directClaim ?p .

            SERVICE wikibase:label {{ bd:serviceParam wikibase:language "es". }}
            }}
            LIMIT 5
            """

            # Ejecutar
            s2 = SPARQLWrapper("https://query.wikidata.org/sparql")
            s2.setQuery(query)
            s2.setReturnFormat(JSON)
            res = s2.query().convert()
            df2_one = pd.json_normalize(res["results"]["bindings"])

            if not df2_one.empty:
                df2_parts.append(df2_one)

        # Concatenamos los resultados de todos los objetos        
        df2 = pd.concat(df2_parts, ignore_index=True) if df2_parts else pd.DataFrame(columns=[
            "s.value","sLabel.value","p.value","prop.value","propLabel.value","o.value","oLabel.value"
        ])

        # Construir tripletas de este nivel
        lvl_triplets = []
        next_frontier = []

        # Se sigue la misma lógica anterior, verificado que exista.
        # Además, hay un filtro adicional para valores "Q123133" que la consulta los devuelve cuando no los encuentra en español
        for _, row in df2.iterrows():
            s_label = row.get("sLabel.value")
            p_label = row.get("propLabel.value")
            o_label = row.get("oLabel.value")

            if s_label and p_label:
                #print(row)
                if not(o_label[0] == "Q" and o_label[1:-1]) and not(s_label[0] == "Q" and s_label[1:-1]):
                    lvl_triplets.append([s_label, p_label, o_label])

            o_uri = str(row.get("o.value", ""))
            if o_uri.startswith("http://www.wikidata.org/entity/"):
                next_frontier.append([s_label, o_uri])

        # Filtramos los duplicados
        df_lvl = (
            pd.DataFrame(lvl_triplets, columns=["entidad", "relacion", "objeto"])
            .drop_duplicates()
        )
        all_triplets.append(df_lvl)

        print("Profundidad", i, "lista\n")

        # Siguiente profundidad, limitamos a extension_limit objetos por entidad
        frontier = []
        count = {}

        for entity, o_uri in next_frontier:
            if entity not in count:
                count[entity] = 1
            else:
                count[entity] += 1

            if count[entity] <= extension_limit:
                frontier.append(o_uri)

    return pd.concat(all_triplets, ignore_index=True).drop_duplicates()

# Entidades a expandir, consideramos máximo 2 objetos por entidad
count = {}
qids_lvl_1 = []

for _, row in df_triplets.iterrows():
    entity = row["entidad"]
    if entity not in count:
        count[entity] = 1
    else:
        count[entity] += 1

    if count[entity] <= 2:
        qids_lvl_1.append(row["uri_object"])

# Llamamos a la función para expandir
df_lvl_2 = expand_level(qids_lvl_1, depth=2)

# Dropeamos los links
df_triplets = df_triplets.drop(["uri_object"], axis=1)

# Dropeamos los duplicados (solo para confirmar)
df_all = pd.concat([df_triplets, df_lvl_2], ignore_index=True).drop_duplicates()
df_all = (
    df_all
    .drop_duplicates(subset=["entidad","relacion","objeto"], keep="first")
)

# Convertimos a dataset
df_all.to_csv("tripletas_mexico.csv", index=False, encoding="utf-8")

print("Archivo guardado")
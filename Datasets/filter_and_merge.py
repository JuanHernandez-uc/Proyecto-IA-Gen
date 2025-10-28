import pandas as pd
from pathlib import Path
import re
import os

def filter_data(df):

    # filtros
    
    ## filtros de entidad
    df = df[~df["entidad"].str.match(r"^Q\d+$", na=False)] # remueve Q432...
    
    ## filtros de relacion
    df = df[df["relacion"] != "se encuentra en el huso horario"]
    
    ## filtros de objeto
    df = df[~df["objeto"].str.match(r"^http", na=False)] # links
    return df
    

carpeta = Path("Datasets")
out_folder = carpeta / "filtered_data"

dfs = []
for f in carpeta.glob("*.csv"):
    if f != "triplets.csv":
        df = pd.read_csv(f)
        df = df[["entidad", "relacion", "objeto"]].copy()
        df["source_file"] = f.name.replace(".csv", "")
        df = filter_data(df)
        out_path = out_folder / f.name
        df.to_csv(out_path, index=False) # guardamos en carpeta filtered_data
        dfs.append(df)

final = pd.concat(dfs, ignore_index=True)
final = final.drop_duplicates(subset=["entidad", "relacion", "objeto"])
final.to_csv(os.path.join(out_folder, "triplets.csv"), index=False)
print(f"Listo. ({len(final)} filas únicas)")
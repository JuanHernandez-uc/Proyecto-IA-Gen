import pandas as pd
from pathlib import Path
import re

def filter_data(df):
    # filtros
    df = df[~df["entidad"].str.match(r"^Q\d+$", na=False)]
    df = df[df["relacion"] != "se encuentra en el huso horario"]
    df = df[~df["objeto"].str.match(r"^http", na=False)]
    return df

carpeta = Path("Datasets")
out_folder = carpeta / "filtered_data"
out_folder.mkdir(parents=True, exist_ok=True)

dfs = []
for f in carpeta.glob("*.csv"):
    if f.name == "triplets.csv":
        continue
    df = pd.read_csv(f)
    df = df[["entidad", "relacion", "objeto"]].copy()
    df["source_file"] = f.stem
    df = filter_data(df)
    (out_folder / f.name).write_text(df.to_csv(index=False))

    dfs.append(df)

final = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["entidad", "relacion", "objeto"])
final.to_csv(out_folder / "triplets.csv", index=False)
print(f"Listo. ({len(final)} filas únicas)")

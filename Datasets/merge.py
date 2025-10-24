import pandas as pd
from pathlib import Path

carpeta = Path("Datasets")

dfs = []
for f in carpeta.glob("*.csv"):
    if f != "triplets.csv":
        df = pd.read_csv(f)
        df = df[["entidad", "relacion", "objeto"]].copy()
        df["source_file"] = f.name.replace(".csv", "")
        dfs.append(df)

final = pd.concat(dfs, ignore_index=True)

final = final.drop_duplicates(subset=["entidad", "relacion", "objeto"])

final.to_csv("Datasets/triplets.csv", index=False)
print(f"Listo. ({len(final)} filas únicas)")
import pandas as pd
import matplotlib.pyplot as plt

def build_long_df(metric_dicts):
    """
    Crea un DataFrame con columnas:
    ['pais', 'modelo', 'metrica', 'valor']
    """
    rows = []
    for metrica, modelos_dict in metric_dicts.items():
        for modelo, scores_por_pais in modelos_dict.items():
            for pais, score in scores_por_pais.items():
                rows.append(
                    {
                        "pais": pais,
                        "modelo": modelo,
                        "metrica": metrica,
                        "valor": float(score),
                    }
                )
    return pd.DataFrame(rows)

def plot_grouped_bars(df_long, metrica, out_dir=None):
    """
    Genera gráfico de barras agrupadas por país para la métrica dada.
    Ordena países de mayor a menor rendimiento promedio entre modelos.
    """
    # filtramos a la métrica que nos interesa
    df_m = df_long[df_long["metrica"] == metrica].copy()

    # pivot: filas = pais, columnas = modelo, valores = score
    wide = df_m.pivot(index="pais", columns="modelo", values="valor")

    # ordenamos países por promedio fila (mejores -> peores)
    wide["avg"] = wide.mean(axis=1)
    wide = wide.sort_values("avg", ascending=False)
    wide = wide.drop(columns=["avg"])

    # plot
    ax = wide.plot(
        kind="bar",
        rot=45,              # rota etiquetas país
        width=0.8,           # deja espacio razonable
        figsize=(10, 5),     # tamaño legible
    )

    ax.set_xlabel("País")
    ax.set_ylabel(metrica)
    ax.set_title(f"{metrica}: comparación Qwen vs Mistral por país")
    ax.legend(title="Modelo")

    plt.tight_layout()
    return ax
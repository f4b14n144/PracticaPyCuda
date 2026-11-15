from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CARPETA = Path(__file__).resolve().parent
GRAFICAS = CARPETA / "graficas"
GRAFICAS.mkdir(exist_ok=True)

VERSIONES = ["c_secuencial", "python_secuencial", "cuda_c", "pycuda"]
ETIQUETAS = {"c_secuencial": "C/C++ secuencial", "python_secuencial": "Python secuencial",
             "cuda_c": "CUDA C", "pycuda": "PyCUDA"}
COLORES = {"c_secuencial": "#2a78d6", "python_secuencial": "#7a4fd6", "cuda_c": "#eb6834", "pycuda": "#1baf7a"}
CLAVES = ["implementacion", "filtro", "resolucion", "mascara", "hilos_por_bloque"]
TIEMPOS = ["t_conjunto", "t_cpu_a_gpu", "t_kernel", "t_gpu_a_cpu", "t_llamada"]
MIN_REPS = 5

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "legend.frameon": False})


def cargar():
    partes = []
    if (CARPETA / "crudo_p01.csv").exists():
        p01 = pd.read_csv(CARPETA / "crudo_p01.csv")
        p01["implementacion"] = p01.implementacion.map({"cpu": "c_secuencial", "cuda": "cuda_c"})
        p01["estado"] = "completada"
        partes.append(p01)
    if (CARPETA / "crudo_python.csv").exists():
        partes.append(pd.read_csv(CARPETA / "crudo_python.csv"))
    if not partes:
        raise SystemExit("No hay crudo_p01.csv ni crudo_python.csv en experimentos/")
    datos = pd.concat(partes, ignore_index=True)
    for t in TIEMPOS:
        if t not in datos:
            datos[t] = np.nan
    return datos


def resumir(crudo):
    filas = []
    for clave, grupo in crudo.groupby(CLAVES):
        medidas = grupo[(grupo.fase == "medicion") & (grupo.estado == "completada")]
        incompleta = (grupo.estado == "no_completada").any() or len(medidas) < MIN_REPS
        fila = dict(zip(CLAVES, clave))
        fila["repeticiones"] = len(medidas)
        fila["estado"] = "no_completada" if incompleta else "completada"
        fila["nota"] = "; ".join(grupo.loc[grupo.estado == "no_completada", "nota"].dropna().astype(str))
        if incompleta:
            cortada = grupo[grupo.estado == "no_completada"]
            fila["t_transcurrido_hasta_corte_ms"] = cortada.t_conjunto.max() if len(cortada) else np.nan
        for t in TIEMPOS:
            v = medidas[t].dropna()
            if incompleta or v.empty:
                continue
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            fila.update({f"{t}_mediana_ms": v.median(), f"{t}_q1_ms": q1, f"{t}_q3_ms": q3,
                         f"{t}_iqr_ms": q3 - q1, f"{t}_desv_ms": v.std(ddof=1),
                         f"{t}_cv_pct": 100 * v.std(ddof=1) / v.mean()})
        filas.append(fila)
    return pd.DataFrame(filas)


def mediana(res, impl, filtro, resol, k, hilos, t="t_conjunto"):
    s = res[(res.implementacion == impl) & (res.filtro == filtro) & (res.resolucion == resol)
            & (res.mascara == k) & (res.hilos_por_bloque == hilos) & (res.estado == "completada")]
    col = f"{t}_mediana_ms"
    return float(s[col].iloc[0]) if len(s) and col in s and pd.notna(s[col].iloc[0]) else np.nan


def aceleraciones(res):
    filas = []
    gpu = res[res.implementacion.isin(["cuda_c", "pycuda"])]
    for _, g in gpu.iterrows():
        sec = "c_secuencial" if g.implementacion == "cuda_c" else "python_secuencial"
        t_sec = mediana(res, sec, g.filtro, g.resolucion, g.mascara, 0)
        if np.isnan(t_sec) or g.estado != "completada":
            continue
        filas.append({"gpu": g.implementacion, "secuencial": sec, "filtro": g.filtro,
                      "resolucion": g.resolucion, "mascara": g.mascara, "hilos_por_bloque": g.hilos_por_bloque,
                      "t_secuencial_ms": t_sec, "t_kernel_ms": g.t_kernel_mediana_ms,
                      "t_conjunto_ms": g.t_conjunto_mediana_ms,
                      "aceleracion_kernel": t_sec / g.t_kernel_mediana_ms,
                      "aceleracion_con_transferencias": t_sec / g.t_conjunto_mediana_ms})
    return pd.DataFrame(filas)


def cuda_vs_pycuda(res):
    filas = []
    for _, p in res[(res.implementacion == "pycuda") & (res.estado == "completada")].iterrows():
        fila = {"filtro": p.filtro, "resolucion": p.resolucion, "mascara": p.mascara,
                "hilos_por_bloque": p.hilos_por_bloque}
        for t in ("t_cpu_a_gpu", "t_kernel", "t_gpu_a_cpu", "t_conjunto"):
            c = mediana(res, "cuda_c", p.filtro, p.resolucion, p.mascara, p.hilos_por_bloque, t)
            fila[f"{t}_cuda_c_ms"] = c
            fila[f"{t}_pycuda_ms"] = p[f"{t}_mediana_ms"]
            fila[f"{t}_razon_pycuda_cuda"] = p[f"{t}_mediana_ms"] / c if c else np.nan
        fila["t_llamada_pycuda_ms"] = p.get("t_llamada_mediana_ms", np.nan)
        filas.append(fila)
    return pd.DataFrame(filas)


def barras(res, casos, etiquetas_x, titulo, archivo, versiones=VERSIONES):
    """Barras agrupadas en escala log. casos: lista de (filtro, resolucion, k)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ancho = 0.8 / len(versiones)
    x = np.arange(len(casos))
    for i, v in enumerate(versiones):
        hilos = 256 if v in ("cuda_c", "pycuda") else 0
        valores = [mediana(res, v, f, r, k, hilos) for f, r, k in casos]
        pos = x + (i - (len(versiones) - 1) / 2) * ancho
        ax.bar(pos, valores, ancho, label=ETIQUETAS[v], color=COLORES[v])
        for p, val in zip(pos, valores):
            if np.isnan(val):
                # Sin barra: configuracion no completada o no medida.
                ax.text(p, 0.02, "no\ncompletada", ha="center", va="bottom", fontsize=7,
                        color=COLORES[v], transform=ax.get_xaxis_transform())
            else:
                ax.text(p, val, f"{val:.3g}", ha="center", va="bottom", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x, etiquetas_x)
    ax.set_ylabel("Tiempo, mediana (ms, escala log)\nsecuencial: convolucion; GPU: transferencias + kernel")
    ax.set_title(titulo, loc="left", fontweight="bold")
    ax.legend(ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.12))
    fig.savefig(GRAFICAS / archivo)
    plt.close(fig)


def grafica_hilos(res):
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, t, nombre in zip(axs, ("t_kernel", "t_conjunto"), ("Kernel", "Transferencias + kernel")):
        for v in ("cuda_c", "pycuda"):
            s = res[(res.implementacion == v) & (res.filtro == "media") & (res.resolucion == 4000)
                    & (res.mascara == 9) & (res.estado == "completada")].sort_values("hilos_por_bloque")
            if s.empty:
                continue
            med = s[f"{t}_mediana_ms"]
            err = [med - s[f"{t}_q1_ms"], s[f"{t}_q3_ms"] - med]
            ax.errorbar(s.hilos_por_bloque.astype(str), med, yerr=err, marker="o", capsize=4,
                        label=ETIQUETAS[v], color=COLORES[v])
        ax.set_title(nombre, loc="left", fontweight="bold")
        ax.set_xlabel("Hilos por bloque")
        ax.set_ylabel("Mediana (ms), barras = IQR")
        ax.legend()
    fig.suptitle("Media 9x9, 4000x4000", x=0.02, ha="left")
    fig.savefig(GRAFICAS / "4_hilos.png")
    plt.close(fig)


def grafica_transferencias(res):
    casos = [("media", 4000, 3), ("nitidez", 4000, 3), ("laplaciano", 4000, 3), ("media", 2000, 9),
             ("media", 4000, 9), ("media", 6000, 9), ("media", 4000, 21)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    etiquetas, x = [], 0
    for f, r, k in casos:
        for v in ("cuda_c", "pycuda"):
            partes = [mediana(res, v, f, r, k, 256, t) for t in ("t_cpu_a_gpu", "t_kernel", "t_gpu_a_cpu")]
            base = 0
            for valor, nombre, color in zip(partes, ("CPU→GPU", "Kernel", "GPU→CPU"),
                                            ("#6250d6", "#1baf7a", "#eda100")):
                if np.isnan(valor):
                    continue
                ax.bar(x, valor, 0.7, bottom=base, color=color, label=nombre if x == 0 else None,
                       hatch="//" if v == "pycuda" else None, edgecolor="white")
                base += valor
            etiquetas.append("C" if v == "cuda_c" else "Py")
            x += 1
        ax.text(x - 1.5, -0.12, f"{f[:3]} {r}\nk={k}", ha="center", va="top", fontsize=8,
                transform=ax.get_xaxis_transform())
        x += 0.5
    posiciones = [i + (i // 2) * 0.5 for i in range(len(etiquetas))]
    ax.set_xticks(posiciones, etiquetas, fontsize=8)
    ax.set_xlabel("C = CUDA C (liso), Py = PyCUDA (rayado)", labelpad=28)
    ax.set_ylabel("Mediana (ms)")
    ax.set_title("Desglose del tiempo GPU (256 hilos por bloque)", loc="left", fontweight="bold")
    ax.legend()
    fig.savefig(GRAFICAS / "5_transferencias.png")
    plt.close(fig)


def grafica_aceleraciones(acel):
    if acel.empty:
        return
    a = acel[acel.hilos_por_bloque == 256].copy()
    a["caso"] = a.filtro.str[:3] + " " + a.resolucion.astype(str) + " k" + a.mascara.astype(str)
    casos = list(dict.fromkeys(a.caso))
    fig, ax = plt.subplots(figsize=(10, 4.2))
    series = [("cuda_c", "aceleracion_kernel", "CUDA C / C++ (kernel)"),
              ("cuda_c", "aceleracion_con_transferencias", "CUDA C / C++ (con transf.)"),
              ("pycuda", "aceleracion_kernel", "PyCUDA / Python (kernel)"),
              ("pycuda", "aceleracion_con_transferencias", "PyCUDA / Python (con transf.)")]
    ancho = 0.2
    for i, (g, col, nombre) in enumerate(series):
        s = a[a.gpu == g].set_index("caso").reindex(casos)[col]
        ax.bar(np.arange(len(casos)) + (i - 1.5) * ancho, s, ancho, label=nombre,
               color=COLORES[g], alpha=1.0 if "kernel" in col else 0.55)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(casos)), casos, fontsize=8)
    ax.set_ylabel("Aceleracion (veces, escala log)")
    ax.set_title("Aceleracion respecto a la version secuencial del mismo lenguaje", loc="left",
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(GRAFICAS / "6_aceleraciones.png")
    plt.close(fig)


def main():
    crudo = cargar()
    res = resumir(crudo)
    res.to_csv(CARPETA / "resumen.csv", index=False, float_format="%.4f")
    acel = aceleraciones(res)
    acel.to_csv(CARPETA / "aceleraciones.csv", index=False, float_format="%.4f")
    cuda_vs_pycuda(res).to_csv(CARPETA / "cuda_vs_pycuda.csv", index=False, float_format="%.4f")

    barras(res, [("media", 4000, 3), ("nitidez", 4000, 3), ("laplaciano", 4000, 3)],
           ["Media 3x3", "Nitidez", "Laplaciano"], "Filtros 3x3, 4000x4000", "1_filtros_4000.png")
    barras(res, [("media", r, 9) for r in (2000, 4000, 6000)], ["2000", "4000", "6000"],
           "Resolucion, media 9x9", "2_resolucion.png")
    barras(res, [("media", 4000, k) for k in (3, 9, 21)], ["3x3", "9x9", "21x21"],
           "Mascara de media, 4000x4000", "3_mascara_4000.png")
    grafica_hilos(res)
    grafica_transferencias(res)
    grafica_aceleraciones(acel)

    incompletas = res[res.estado == "no_completada"]
    if len(incompletas):
        print("Configuraciones NO completadas (no entran en aceleraciones):")
        print(incompletas[CLAVES + ["repeticiones", "nota"]].to_string(index=False))
    print(f"Resumen en {CARPETA / 'resumen.csv'}; graficas en {GRAFICAS}")


if __name__ == "__main__":
    main()

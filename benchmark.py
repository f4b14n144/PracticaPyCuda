import argparse
import csv
import datetime as dt
import sys

import numpy as np
import time
from pathlib import Path

from filtros.comun import LAPLACIANO, MEDIA, NITIDEZ, NOMBRES, RAIZ, cargar_gris, ruta_imagen, verificar_sha256
from filtros import secuencial
from filtros.secuencial import Interrumpido

CAMPOS = ["implementacion", "experimentos", "filtro", "resolucion", "mascara", "hilos_por_bloque", "bloques",
          "fase", "repeticion", "t_cpu_a_gpu", "t_kernel", "t_gpu_a_cpu", "t_conjunto", "t_llamada",
          "unidad", "estado", "nota"]

# (experimentos, filtro, resolucion, mascara, hilos). Iguales a benchmark.cpp de la practica 01.
SECUENCIAL = [
    ("filtro;mascara", MEDIA, 4000, 3, 0),
    ("filtro", NITIDEZ, 4000, 3, 0),
    ("filtro", LAPLACIANO, 4000, 3, 0),
    ("resolucion", MEDIA, 2000, 9, 0),
    ("resolucion;mascara;hilos", MEDIA, 4000, 9, 0),
    ("resolucion", MEDIA, 6000, 9, 0),
    ("mascara", MEDIA, 4000, 21, 0),
]
GPU = [(e, f, r, k, 256) for e, f, r, k, _ in SECUENCIAL] + [
    ("hilos", MEDIA, 4000, 9, 128),
    ("hilos", MEDIA, 4000, 9, 512),
]


def clave(impl, f, r, k, h):
    return (impl, NOMBRES[f], str(r), str(k), str(h))


def configs_terminadas(ruta, reps):
    """Configuraciones que ya tienen todas sus mediciones o quedaron no completadas."""
    if not ruta.exists():
        return set()
    conteo, cortadas = {}, set()
    with open(ruta, newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            c = (fila["implementacion"], fila["filtro"], fila["resolucion"], fila["mascara"],
                 fila["hilos_por_bloque"])
            if fila["estado"] == "no_completada":
                cortadas.add(c)
            elif fila["fase"] == "medicion":
                conteo[c] = conteo.get(c, 0) + 1
    hechas = {c for c, n in conteo.items() if n >= reps.get(c[0], 0)}
    return hechas | cortadas


def estimar_secuencial(img, f, k, lado=200):
    """Prueba pequena: tiempo por pixel en un recorte, para estimar la imagen completa.

    Es solo una ESTIMACION para decidir el orden y el limite; nunca se registra como
    tiempo medido.
    """
    recorte = img[:lado, :lado].tobytes()
    secuencial.aplicar_filtro(recorte, lado, lado, f, k)        # calentamiento
    t0 = time.perf_counter()
    secuencial.aplicar_filtro(recorte, lado, lado, f, k)
    por_pixel = (time.perf_counter() - t0) / (lado * lado)
    return por_pixel * img.size


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--versiones", default="pycuda,python", help="pycuda, python o ambas separadas por coma")
    p.add_argument("--reps-gpu", type=int, default=10)
    p.add_argument("--calent-gpu", type=int, default=2)
    p.add_argument("--reps-python", type=int, default=5)
    p.add_argument("--calent-python", type=int, default=1)
    p.add_argument("--limite-min", type=float, default=30.0,
                   help="tiempo maximo de UNA ejecucion de Python secuencial antes de interrumpirla")
    p.add_argument("--csv", default=str(RAIZ / "experimentos" / "crudo_python.csv"))
    p.add_argument("--reanudar", action="store_true", help="agrega al CSV y salta lo ya terminado")
    p.add_argument("--solo-estimar", action="store_true")
    args = p.parse_args()
    versiones = {v.strip() for v in args.versiones.split(",")}

    for nombre, ok in verificar_sha256().items():
        if not ok:
            print(f"{nombre} no coincide con SHA256SUMS.txt: las entradas no son las de la practica 01")
            return 1
    # Las imagenes se cargan una sola vez, antes de medir.
    imagenes = {r: cargar_gris(ruta_imagen(r)) for r in (2000, 4000, 6000)}

    ruta = Path(args.csv)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    reps = {"pycuda": args.reps_gpu, "python_secuencial": args.reps_python}
    hechas = configs_terminadas(ruta, reps) if args.reanudar else set()
    archivo = escritor = None
    if not args.solo_estimar:
        # Sin --reanudar se borran solo las filas de las versiones que se van a medir;
        # las de las otras versiones se conservan.
        conservar = []
        if ruta.exists() and not args.reanudar:
            medir = {"pycuda" if v == "pycuda" else "python_secuencial" for v in versiones}
            with open(ruta, newline="", encoding="utf-8") as previo:
                conservar = [f for f in csv.DictReader(previo) if f["implementacion"] not in medir]
        nuevo = not ruta.exists() or not args.reanudar
        archivo = open(ruta, "w" if nuevo else "a", newline="", encoding="utf-8")
        escritor = csv.DictWriter(archivo, fieldnames=CAMPOS)
        if nuevo:
            escritor.writeheader()
            escritor.writerows(conservar)

    def fila(impl, exp, f, r, k, h, bloques, fase, rep, estado="completada", nota="", **t):
        escritor.writerow({"implementacion": impl, "experimentos": exp, "filtro": NOMBRES[f], "resolucion": r,
                           "mascara": k, "hilos_por_bloque": h, "bloques": bloques, "fase": fase,
                           "repeticion": rep, "unidad": "ms", "estado": estado, "nota": nota,
                           **{c: (f"{t[c]:.6f}" if c in t else "") for c in
                              ("t_cpu_a_gpu", "t_kernel", "t_gpu_a_cpu", "t_conjunto", "t_llamada")}})
        archivo.flush()

    try:
        if "pycuda" in versiones and not args.solo_estimar:
            from filtros.gpu import FiltrosGPU
            with FiltrosGPU() as gpu:
                prep = RAIZ / "experimentos" / "preparacion.csv"
                existe = prep.exists()
                with open(prep, "a", newline="", encoding="utf-8") as pa:
                    w = csv.writer(pa)
                    if not existe:
                        w.writerow(["fecha", "gpu", "t_contexto_ms", "t_compilacion_ms", "cache_compilacion"])
                    w.writerow([dt.datetime.now().isoformat(timespec="seconds"), gpu.nombre,
                                f"{gpu.t_contexto_ms:.3f}", f"{gpu.t_compilacion_ms:.3f}",
                                "si" if gpu.usar_cache else "no"])
                print(f"GPU {gpu.nombre}: contexto {gpu.t_contexto_ms:.1f} ms, "
                      f"compilacion {gpu.t_compilacion_ms:.1f} ms, maximo {gpu.max_hilos} hilos/bloque")
                for exp, f, r, k, h in GPU:
                    if clave("pycuda", f, r, k, h) in hechas:
                        continue
                    if h > gpu.max_hilos:
                        print(f"  la GPU no permite {h} hilos por bloque; se omite")
                        continue
                    print(f"PyCUDA {NOMBRES[f]} {r} k={k} hilos={h}", flush=True)
                    salida = np.empty((r, r), dtype=np.float32)   # se reutiliza, como en benchmark.cpp
                    total = args.calent_gpu + args.reps_gpu
                    for i in range(1, total + 1):
                        _, t = gpu.ejecutar(imagenes[r], f, k, h, salida=salida)
                        calentando = i <= args.calent_gpu
                        fila("pycuda", exp, f, r, k, h, t["bloques"],
                             "calentamiento" if calentando else "medicion",
                             i if calentando else i - args.calent_gpu,
                             **{c: t[c] for c in ("t_cpu_a_gpu", "t_kernel", "t_gpu_a_cpu",
                                                  "t_conjunto", "t_llamada")})

        if "python" in versiones:
            limite_s = args.limite_min * 60
            # La version secuencial recibe bytes (sin NumPy); se preparan antes de medir.
            entradas = {r: secuencial.cargar_gris(ruta_imagen(r))[0] for r in imagenes}
            pendientes = []
            print("\nPrueba pequena de viabilidad de Python secuencial (recorte 200x200):")
            for exp, f, r, k, h in SECUENCIAL:
                if clave("python_secuencial", f, r, k, h) in hechas:
                    continue
                est = estimar_secuencial(imagenes[r], f, k)
                corridas = args.calent_python + args.reps_python
                print(f"  {NOMBRES[f]:10s} {r} k={k:2d}: ~{est / 60:7.1f} min por ejecucion, "
                      f"~{est * corridas / 3600:5.1f} h con {corridas} ejecuciones (ESTIMADO)"
                      + ("  -> superaria el limite" if est > limite_s else ""))
                pendientes.append((est, exp, f, r, k, h))
            if args.solo_estimar:
                return 0
            pendientes.sort()                                  # primero las mas cortas
            lado = 200
            for _, exp, f, r, k, h in pendientes:
                print(f"Python secuencial {NOMBRES[f]} {r} k={k}", flush=True)
                # Calentamiento sobre un recorte: en Python puro solo prepara el interprete
                # (no hay compilacion ni cache de GPU), asi que no hace falta la imagen completa.
                recorte = imagenes[r][:lado, :lado].tobytes()
                for i in range(1, args.calent_python + 1):
                    t0 = time.perf_counter()
                    secuencial.aplicar_filtro(recorte, lado, lado, f, k)
                    fila("python_secuencial", exp, f, r, k, h, 0, "calentamiento", i,
                         nota=f"sobre recorte {lado}x{lado}", t_conjunto=(time.perf_counter() - t0) * 1000)
                for rep in range(1, args.reps_python + 1):
                    fase = "medicion"
                    t0 = time.perf_counter()
                    try:
                        secuencial.aplicar_filtro(entradas[r], r, r, f, k, limite_s=limite_s)
                    except Interrumpido as e:
                        nota = (f"interrumpida al superar el limite de {args.limite_min:g} min; "
                                f"{e.filas}/{e.alto} filas procesadas")
                        print(f"  NO COMPLETADA: {nota}")
                        fila("python_secuencial", exp, f, r, k, h, 0, fase, rep, "no_completada", nota,
                             t_conjunto=e.transcurrido_s * 1000)
                        break
                    ms = (time.perf_counter() - t0) * 1000
                    print(f"  medicion {rep}: {ms / 1000:.1f} s", flush=True)
                    fila("python_secuencial", exp, f, r, k, h, 0, fase, rep, t_conjunto=ms)
    finally:
        if archivo:
            archivo.close()
    print(f"Registros en {ruta}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

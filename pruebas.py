import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

from filtros.comun import (LAPLACIANO, MEDIA, NITIDEZ, NOMBRES, RAIZ, TOLERANCIA, cargar_gris,
                           cargar_referencia, convertir_a_visualizacion, errores, guardar_gris,
                           imagen_lcg, ruta_imagen, verificar_sha256)
from filtros import secuencial

HILOS = (128, 256, 512)
CASOS = ((MEDIA, 3), (MEDIA, 9), (MEDIA, 21), (NITIDEZ, 3), (LAPLACIANO, 3))
CENTINELA = np.nan   # valor imposible en la salida: detecta pixeles no escritos

fallos = 0
filas_csv = []


def python_secuencial(img, f, k):
    """Ejecuta la version secuencial (sin NumPy) y pasa su salida a NumPy solo para
    calcular errores."""
    alto, ancho = img.shape
    salida = secuencial.aplicar_filtro(img.tobytes(), ancho, alto, f, k)
    return np.frombuffer(salida, dtype=np.float32).reshape(alto, ancho)


def comprobar(ok, nombre):
    global fallos
    print(("[OK]    " if ok else "[FALLA] ") + nombre, flush=True)
    if not ok:
        fallos += 1
    return ok


def registrar(prueba, version, referencia, filtro, ancho, alto, k, hilos, ref, valores):
    sin_escribir = int(np.isnan(valores).sum())
    emax, emed = errores(ref, valores) if sin_escribir == 0 else (float("nan"), float("nan"))
    ok = sin_escribir == 0 and emax <= TOLERANCIA
    pixeles = ancho * alto
    filas_csv.append({
        "prueba": prueba, "version": version, "referencia": referencia, "filtro": NOMBRES[filtro],
        "ancho": ancho, "alto": alto, "mascara": k, "hilos_por_bloque": hilos,
        "bloques": (pixeles + hilos - 1) // hilos if hilos else 0,
        "resto_pixeles_entre_hilos": pixeles % hilos if hilos else 0,
        "error_abs_max": emax, "error_abs_medio": emed, "pixeles_sin_escribir": sin_escribir,
        "tolerancia": TOLERANCIA, "cumple": "si" if ok else "no",
    })
    comprobar(ok, f"{prueba} {version} {NOMBRES[filtro]} {ancho}x{alto} k={k} hilos={hilos} "
                  f"errMax={emax:.3g} sinEscribir={sin_escribir}")


def versiones_c(etiqueta, filtro, k, alto, ancho, hilos_lista):
    """Salidas de C/C++ y CUDA C exportadas por exportar_salidas.exe (o None)."""
    cpu = cargar_referencia(etiqueta, filtro, k, "cpu", alto, ancho)
    cuda = {h: cargar_referencia(etiqueta, filtro, k, "cuda", alto, ancho, h) for h in hilos_lista}
    return cpu, {h: v for h, v in cuda.items() if v is not None}


def prueba_manual(gpu):
    print("\n== Matriz pequena calculada a mano ==")
    m = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    esperados = {
        MEDIA: np.array([120, 210, 160, 270, 450, 330, 240, 390, 280], dtype=np.float32) / np.float32(9),
        NITIDEZ: np.array([-10, 10, 70, 70, 50, 130, 230, 190, 310], dtype=np.float32),
        LAPLACIANO: np.array([20, 10, -40, -30, 0, -70, -160, -110, -220], dtype=np.float32),
    }
    for f, esperado in esperados.items():
        esperado = esperado.reshape(3, 3)
        registrar("manual", "python_secuencial", "a_mano", f, 3, 3, 3, 0, esperado, python_secuencial(m, f, 3))
        if gpu:
            for h in HILOS:
                s, _ = gpu.ejecutar(m, f, 3, h, centinela=CENTINELA)
                registrar("manual", "pycuda", "a_mano", f, 3, 3, 3, h, esperado, s)
        cpu, cuda = versiones_c("manual", f, 3, 3, 3, HILOS)
        if cpu is not None:
            registrar("manual", "c_secuencial", "a_mano", f, 3, 3, 3, 0, esperado, cpu)
        for h, v in cuda.items():
            registrar("manual", "cuda_c", "a_mano", f, 3, 3, 3, h, esperado, v)

    print("\n== Conversion a 0-255 ==")
    casos = [(-10, NITIDEZ, 0), (310, NITIDEZ, 255), (-220, LAPLACIANO, 0), (-30, LAPLACIANO, 98),
             (200, LAPLACIANO, 255), (120 / 9, MEDIA, 13), (160 / 9, MEDIA, 18)]
    for valor, f, esperado in casos:
        numpy_ = int(convertir_a_visualizacion(np.array([valor], dtype=np.float32), f)[0])
        puro = secuencial.convertir_a_visualizacion(np.array([valor], dtype=np.float32).tolist(), f)[0]
        comprobar(numpy_ == esperado and puro == esperado,
                  f"{NOMBRES[f]} {valor:.2f} -> {esperado} (NumPy {numpy_}, Python puro {puro})")


def prueba_no_multiplos(gpu):
    print("\n== Dimensiones no multiplos de los hilos por bloque ==")
    for ancho, alto in ((1, 1), (7, 5), (513, 1), (1, 257), (1001, 777)):
        img = imagen_lcg(ancho, alto)
        etiqueta = f"lcg{ancho}x{alto}"
        for f, k in CASOS:
            sec = python_secuencial(img, f, k)
            cpu, cuda = versiones_c(etiqueta, f, k, alto, ancho, HILOS)
            if cpu is None:
                print(f"  (sin referencia C/C++ para {etiqueta}; se usa Python secuencial)")
                ref, nombre_ref = sec, "python_secuencial"
            else:
                ref, nombre_ref = cpu, "c_secuencial"
                registrar("no_multiplo", "python_secuencial", nombre_ref, f, ancho, alto, k, 0, ref, sec)
            for h, v in cuda.items():
                registrar("no_multiplo", "cuda_c", nombre_ref, f, ancho, alto, k, h, ref, v)
            if gpu:
                for h in HILOS:
                    s, _ = gpu.ejecutar(img, f, k, h, centinela=CENTINELA)
                    registrar("no_multiplo", "pycuda", nombre_ref, f, ancho, alto, k, h, ref, s)


def prueba_imagenes(gpu, completo):
    print("\n== Imagenes reales contra C/C++ secuencial ==")
    configs = [(f, 2000, k, 256) for f, k in CASOS]
    configs += [(f, 4000, k, 256) for f, k in CASOS]
    configs += [(MEDIA, 4000, 9, 128), (MEDIA, 4000, 9, 512), (MEDIA, 6000, 9, 256)]
    imagenes = {}
    for f, res, k, h in configs:
        if res not in imagenes:
            imagenes[res] = cargar_gris(ruta_imagen(res))
            comprobar(imagenes[res].shape == (res, res), f"{res}.png tiene el tamano esperado")
        img = imagenes[res]
        etiqueta = f"img{res}"
        cpu, cuda = versiones_c(etiqueta, f, k, res, res, (h,))
        if cpu is None:
            comprobar(False, f"falta la referencia C/C++ {etiqueta} {NOMBRES[f]} k={k}: "
                             "ejecutar exportar_salidas.exe de la practica 01")
            continue
        for hh, v in cuda.items():
            registrar("imagen", "cuda_c", "c_secuencial", f, res, res, k, hh, cpu, v)
        if gpu:
            s, _ = gpu.ejecutar(img, f, k, h, centinela=CENTINELA)
            registrar("imagen", "pycuda", "c_secuencial", f, res, res, k, h, cpu, s)
        python_viable = h == 256 and (res == 2000 or (completo and res == 4000 and k == 3))
        if python_viable:
            t0 = time.perf_counter()
            sec = python_secuencial(img, f, k)
            print(f"  Python secuencial {NOMBRES[f]} {res} k={k}: {time.perf_counter() - t0:.1f} s")
            registrar("imagen", "python_secuencial", "c_secuencial", f, res, res, k, 0, cpu, sec)


def evidencia_visual(gpu, carpeta):
    """Original y los tres filtros en 2000x2000, mas un recorte de la misma region."""
    print("\n== Evidencia visual ==")
    img = cargar_gris(ruta_imagen(2000))
    y0, x0, lado = 900, 900, 300               # misma region en todas las imagenes
    recortes = [img[y0:y0 + lado, x0:x0 + lado]]
    guardar_gris(carpeta / "original_2000.png", img)
    for f, k, nombre in ((MEDIA, 3, "media_3"), (MEDIA, 21, "media_21"),
                         (NITIDEZ, 3, "nitidez"), (LAPLACIANO, 3, "laplaciano")):
        if gpu:
            salida, _ = gpu.ejecutar(img, f, k, 256)
        else:
            salida = cargar_referencia("img2000", f, k, "cpu", 2000, 2000)
            if salida is None:
                continue
        vis = convertir_a_visualizacion(salida, f)
        guardar_gris(carpeta / f"{nombre}_2000.png", vis)
        recortes.append(vis[y0:y0 + lado, x0:x0 + lado])
    separador = np.full((lado, 8), 255, dtype=np.uint8)
    mosaico = np.hstack([p for r in recortes for p in (r, separador)][:-1])
    guardar_gris(carpeta / "recortes_2000.png", mosaico)
    comprobar(True, f"imagenes guardadas en {carpeta} (recorte {lado}x{lado} desde ({x0},{y0}); "
                    "orden: original, media 3, media 21, nitidez, laplaciano)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sin-gpu", action="store_true", help="no usar PyCUDA")
    p.add_argument("--completo", action="store_true", help="Python secuencial tambien en 4000x4000 (3x3)")
    p.add_argument("--salida", default=str(RAIZ / "experimentos"))
    args = p.parse_args()
    salida = Path(args.salida)
    salida.mkdir(parents=True, exist_ok=True)

    print("== Entradas identicas a la practica 01 (SHA-256) ==")
    for nombre, ok in verificar_sha256().items():
        comprobar(ok, f"{nombre} coincide con SHA256SUMS.txt")

    gpu = None
    if not args.sin_gpu:
        from filtros.gpu import FiltrosGPU
        gpu = FiltrosGPU()
        print(f"GPU: {gpu.nombre}, contexto {gpu.t_contexto_ms:.1f} ms, "
              f"compilacion {gpu.t_compilacion_ms:.1f} ms")
    try:
        prueba_manual(gpu)
        prueba_no_multiplos(gpu)
        prueba_imagenes(gpu, args.completo)
        evidencia_visual(gpu, salida / "imagenes")
    finally:
        if gpu:
            gpu.cerrar()

    with open(salida / "validacion.csv", "w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas_csv[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas_csv)
    print(f"\nRegistros en {salida / 'validacion.csv'}")
    print("TODAS LAS PRUEBAS PASARON" if fallos == 0 else f"HAY FALLOS: {fallos}")
    return 0 if fallos == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

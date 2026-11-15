import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
ENTRADA = RAIZ / "Imagenes" / "entrada"
REFERENCIAS = RAIZ / "referencias"   # salidas de exportar_salidas.exe (practica 01)

from .constantes import LAPLACIANO, MASCARA_LAPLACIANO, MASCARA_NITIDEZ, MEDIA, NITIDEZ, NOMBRES  # noqa: F401

# Tolerancia sobre la salida float32 (ver README, seccion Validacion).
TOLERANCIA = 1e-4


def mascara_media(k):
    """Media k x k: todos los coeficientes valen 1/(k*k).

    En las cuatro versiones se suman los k*k vecinos como enteros y se divide una
    sola vez entre k*k, que es lo mismo que multiplicar por 1/(k*k) sin acumular
    errores de redondeo en cada producto.
    """
    return np.full((k, k), 1.0 / (k * k), dtype=np.float32)


def cargar_gris(ruta):
    """Lee un PNG de 8 bits en escala de grises como arreglo uint8 contiguo (fila mayor)."""
    with Image.open(ruta) as img:
        if img.mode != "L":
            # Las entradas de la practica ya estan en modo L; se convierte con ITU-R 601
            # (la misma formula de ResizeImage.py) solo si se usa otra imagen.
            img = img.convert("L")
        datos = np.asarray(img, dtype=np.uint8)
    return np.ascontiguousarray(datos)


def guardar_gris(ruta, datos_uint8):
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(datos_uint8, mode="L").save(ruta)


def ruta_imagen(resolucion):
    return ENTRADA / f"{resolucion}.png"


def verificar_sha256(carpeta=ENTRADA):
    """Comprueba que las entradas son exactamente las de la practica 01."""
    resultados = {}
    for linea in (Path(carpeta) / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        if not linea.strip():
            continue
        esperado, nombre = linea.split()
        ruta = Path(carpeta) / nombre
        real = hashlib.sha256(ruta.read_bytes()).hexdigest() if ruta.exists() else None
        resultados[nombre] = real == esperado
    return resultados


def convertir_a_visualizacion(salida, filtro):
    """Igual que convertirAVisualizacion() de filtros.cpp, aplicado a todo el arreglo.

    Media: redondeo. Nitidez: saturacion a [0, 255]. Laplaciano: +128 y saturacion.
    std::round redondea .5 alejandose de cero; np.round lo haria al par, por eso se
    usa floor(|v| + 0.5) con signo.
    """
    v = salida.astype(np.float32, copy=True)
    if filtro == LAPLACIANO:
        v += np.float32(128.0)
    v = np.sign(v) * np.floor(np.abs(v) + np.float32(0.5))
    return np.clip(v, 0, 255).astype(np.uint8)


def errores(referencia, valores):
    """Error absoluto maximo y medio, calculados en float64 antes de convertir a 0-255."""
    d = np.abs(referencia.astype(np.float64) - valores.astype(np.float64))
    return float(d.max()), float(d.mean())


def imagen_lcg(ancho, alto):
    """Misma imagen pseudoaleatoria que pruebas.cpp (generador lineal congruente fijo)."""
    estado = 12345
    datos = bytearray(ancho * alto)
    for i in range(ancho * alto):
        estado = (estado * 1103515245 + 12345) & 0xFFFFFFFF
        datos[i] = (estado >> 16) & 0xFF
    return np.frombuffer(bytes(datos), dtype=np.uint8).reshape(alto, ancho).copy()


def nombre_referencia(etiqueta, filtro, k, version, hilos=None):
    """Nombre del archivo .f32 que escribe exportar_salidas.cpp (repo de la practica 01)."""
    sufijo = version if hilos is None else f"{version}{hilos}"
    return REFERENCIAS / f"{etiqueta}_{NOMBRES[filtro]}_k{k}_{sufijo}.f32"


def cargar_referencia(etiqueta, filtro, k, version, alto, ancho, hilos=None):
    ruta = nombre_referencia(etiqueta, filtro, k, version, hilos)
    if not ruta.exists():
        return None
    datos = np.fromfile(ruta, dtype=np.float32)
    if datos.size != ancho * alto:
        raise ValueError(f"{ruta} tiene {datos.size} valores, se esperaban {ancho * alto}")
    return datos.reshape(alto, ancho)

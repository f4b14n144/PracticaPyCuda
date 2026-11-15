import math
import time
from array import array

from PIL import Image

from .constantes import LAPLACIANO, MASCARA_LAPLACIANO, MASCARA_NITIDEZ, MEDIA, NITIDEZ


class Interrumpido(Exception):
    """La ejecucion supero el limite de tiempo; se conservan la evidencia y el progreso."""

    def __init__(self, transcurrido_s, filas, alto):
        super().__init__(f"interrumpido tras {transcurrido_s:.1f} s ({filas}/{alto} filas)")
        self.transcurrido_s = transcurrido_s
        self.filas = filas
        self.alto = alto


def cargar_gris(ruta):
    """Devuelve (bytes, ancho, alto) de un PNG en escala de grises."""
    with Image.open(ruta) as img:
        if img.mode != "L":
            img = img.convert("L")
        return img.tobytes(), img.width, img.height


def guardar_gris(ruta, datos, ancho, alto):
    Image.frombytes("L", (ancho, alto), bytes(datos)).save(ruta)


def _media(entrada, ancho, alto, k, salida, limite_s, t0):
    radio = k // 2
    kk = k * k
    desplazamientos = range(-radio, radio + 1)
    for y in range(alto):
        if limite_s is not None and time.perf_counter() - t0 > limite_s:
            raise Interrumpido(time.perf_counter() - t0, y, alto)
        base = y * ancho                                 # inicio de la fila de salida
        for x in range(ancho):
            suma = 0
            for ky in desplazamientos:                   # filas de la mascara
                yy = y + ky
                if yy < 0 or yy >= alto:
                    continue                             # fila fuera: sus vecinos valen 0
                fila = yy * ancho                        # inicio de la fila vecina
                for kx in desplazamientos:               # columnas de la mascara
                    xx = x + kx
                    if 0 <= xx < ancho:
                        suma += entrada[fila + xx]
            salida[base + x] = suma / kk


def _mascara3x3(entrada, ancho, alto, mascara, salida, limite_s, t0):
    for y in range(alto):
        if limite_s is not None and time.perf_counter() - t0 > limite_s:
            raise Interrumpido(time.perf_counter() - t0, y, alto)
        base = y * ancho
        for x in range(ancho):
            suma = 0
            for ky in (-1, 0, 1):
                yy = y + ky
                if yy < 0 or yy >= alto:
                    continue
                fila = yy * ancho
                coeficientes = mascara[ky + 1]           # fila de la mascara
                for kx in (-1, 0, 1):
                    xx = x + kx
                    if 0 <= xx < ancho:
                        suma += coeficientes[kx + 1] * entrada[fila + xx]
            salida[base + x] = suma


def aplicar_filtro(entrada, ancho, alto, filtro, k=3, limite_s=None):
    """Aplica el filtro y devuelve array('f') de ancho * alto valores.

    limite_s: si la convolucion pasa de este tiempo se lanza Interrumpido.
    """
    if len(entrada) != ancho * alto:
        raise ValueError("la entrada no tiene ancho * alto pixeles")
    salida = array("f", bytes(4 * ancho * alto))          # ancho * alto ceros float32
    t0 = time.perf_counter()
    if filtro == MEDIA:
        if k < 1 or k % 2 == 0:
            raise ValueError("la mascara de media debe ser impar")
        _media(entrada, ancho, alto, k, salida, limite_s, t0)
    elif filtro == NITIDEZ:
        _mascara3x3(entrada, ancho, alto, MASCARA_NITIDEZ, salida, limite_s, t0)
    elif filtro == LAPLACIANO:
        _mascara3x3(entrada, ancho, alto, MASCARA_LAPLACIANO, salida, limite_s, t0)
    else:
        raise ValueError(f"filtro desconocido: {filtro}")
    return salida


def convertir_a_visualizacion(salida, filtro):
    """Igual que convertirAVisualizacion() de filtros.cpp, pixel por pixel.

    Media: redondeo (como std::round, .5 se aleja de cero). Nitidez: saturacion a
    [0, 255]. Laplaciano: +128 y saturacion.
    """
    resultado = bytearray(len(salida))
    for i, valor in enumerate(salida):
        if filtro == LAPLACIANO:
            valor += 128.0
        v = math.copysign(math.floor(abs(valor) + 0.5), valor)
        resultado[i] = 0 if v < 0 else 255 if v > 255 else int(v)
    return resultado

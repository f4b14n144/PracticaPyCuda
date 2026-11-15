import glob
import os
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

from .comun import LAPLACIANO, MEDIA, NITIDEZ

RUTA_KERNEL = Path(__file__).resolve().parent / "kernels" / "filtros_pycuda.cu"


def asegurar_cl_en_path():
    """En Windows nvcc necesita cl.exe (MSVC). Si no esta en el PATH (por ejemplo al
    ejecutar desde PyCharm), se busca con vswhere y se agrega al PATH del proceso."""
    if sys.platform != "win32" or shutil.which("cl"):
        return
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return
    instalacion = subprocess.run(
        [str(vswhere), "-latest", "-products", "*", "-requires",
         "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"],
        capture_output=True, text=True).stdout.strip()
    candidatos = sorted(glob.glob(os.path.join(instalacion, "VC", "Tools", "MSVC", "*", "bin", "Hostx64", "x64")))
    if candidatos:
        os.environ["PATH"] = candidatos[-1] + os.pathsep + os.environ["PATH"]


def _ms(a, b):
    return (b - a) * 1000.0


class FiltrosGPU:
    def __init__(self, dispositivo=0, usar_cache=False):
        # 1) Contexto CUDA (equivale a la primera llamada CUDA del programa en C).
        t0 = time.perf_counter()
        cuda.init()
        self.dispositivo = cuda.Device(dispositivo)
        self.contexto = self.dispositivo.make_context()
        t1 = time.perf_counter()
        self.t_contexto_ms = _ms(t0, t1)

        # 2) Compilacion con nvcc y carga del modulo. Con usar_cache=False se compila
        #    siempre, para medir la compilacion real y no una lectura de la cache.
        asegurar_cl_en_path()
        fuente = RUTA_KERNEL.read_text(encoding="utf-8")
        t2 = time.perf_counter()
        try:
            # En Windows nvcc imprime el nombre del archivo (kernel.cu); no es un error.
            warnings.filterwarnings("ignore", message="The CUDA compiler succeeded")
            self.modulo = SourceModule(
                fuente,
                no_extern_c=True,                  # el .cu ya declara extern "C"
                include_dirs=[str(RUTA_KERNEL.parent)],   # filtros.h de la practica 01
                cache_dir=None if usar_cache else False,
            )
        except Exception:
            self.cerrar()                          # no dejar el contexto activo
            raise
        self.kernels = {
            MEDIA: self.modulo.get_function("kernelFiltroMedia"),
            NITIDEZ: self.modulo.get_function("kernelFiltroNitidez"),
            LAPLACIANO: self.modulo.get_function("kernelFiltroLaplaciano"),
        }
        # prepare() fija los tipos de los argumentos: P = puntero, i = int de 32 bits.
        self.kernels[MEDIA].prepare("PPiii")
        self.kernels[NITIDEZ].prepare("PPii")
        self.kernels[LAPLACIANO].prepare("PPii")
        t3 = time.perf_counter()
        self.t_compilacion_ms = _ms(t2, t3)
        self.usar_cache = usar_cache

        self.max_hilos = self.dispositivo.get_attribute(cuda.device_attribute.MAX_THREADS_PER_BLOCK)
        self.nombre = self.dispositivo.name()
        self.memoria_mib = self.dispositivo.total_memory() // (1024 * 1024)

    def cerrar(self):
        if self.contexto is not None:
            self.contexto.pop()
            self.contexto.detach()
            self.contexto = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cerrar()

    def ejecutar(self, entrada, filtro, k=3, hilos_por_bloque=256, centinela=None, salida=None):
        """Aplica el filtro en la GPU. Devuelve (salida float32 (alto, ancho), tiempos).

        centinela: si se da, la salida en la GPU se llena con ese valor antes del
        kernel. Sirve para comprobar que todos los pixeles fueron escritos.
        salida: arreglo float32 (alto, ancho) ya reservado para reutilizarlo entre
        repeticiones, igual que benchmark.cpp reutiliza su std::vector. Un arreglo
        nuevo en cada llamada haria que GPU->CPU incluya la asignacion de paginas.
        """
        if entrada.dtype != np.uint8 or entrada.ndim != 2:
            raise TypeError("la entrada debe ser uint8 de 2 dimensiones")
        if not entrada.flags.c_contiguous:
            raise ValueError("la entrada debe ser contigua en memoria (fila mayor)")
        if not 1 <= hilos_por_bloque <= self.max_hilos:
            raise ValueError(f"la GPU no permite {hilos_por_bloque} hilos por bloque")

        alto, ancho = entrada.shape
        pixeles = ancho * alto
        if salida is None:
            salida = np.empty((alto, ancho), dtype=np.float32)
        elif salida.shape != (alto, ancho) or salida.dtype != np.float32 or not salida.flags.c_contiguous:
            raise ValueError("salida debe ser float32 contigua del tamano de la entrada")
        bloques = (pixeles + hilos_por_bloque - 1) // hilos_por_bloque

        t_llamada0 = time.perf_counter()
        # Reserva: fuera del tiempo conjunto, igual que en CUDA C.
        d_entrada = cuda.mem_alloc(entrada.nbytes)            # 1 byte por pixel
        d_salida = cuda.mem_alloc(salida.nbytes)              # 4 bytes por pixel
        if centinela is not None:
            bits = int(np.array([centinela], dtype=np.float32).view(np.uint32)[0])
            cuda.memset_d32(d_salida, bits, pixeles)
        inicio, fin = cuda.Event(), cuda.Event()
        self.contexto.synchronize()

        t0 = time.perf_counter()
        cuda.memcpy_htod(d_entrada, entrada)
        self.contexto.synchronize()
        t1 = time.perf_counter()

        kernel = self.kernels[filtro]
        malla = (bloques, 1)
        bloque = (hilos_por_bloque, 1, 1)
        inicio.record()
        if filtro == MEDIA:
            kernel.prepared_call(malla, bloque, d_entrada, d_salida,
                                 np.int32(ancho), np.int32(alto), np.int32(k))
        else:
            kernel.prepared_call(malla, bloque, d_entrada, d_salida,
                                 np.int32(ancho), np.int32(alto))
        fin.record()
        fin.synchronize()                                     # antes de consultar el tiempo
        t2 = time.perf_counter()

        cuda.memcpy_dtoh(salida, d_salida)
        self.contexto.synchronize()
        t3 = time.perf_counter()

        kernel_ms = fin.time_since(inicio)                    # eventos CUDA, en ms
        d_entrada.free()
        d_salida.free()
        t_llamada1 = time.perf_counter()

        tiempos = {
            "t_cpu_a_gpu": _ms(t0, t1),
            "t_kernel": kernel_ms,
            "t_gpu_a_cpu": _ms(t2, t3),
            "t_conjunto": _ms(t0, t3),
            "t_llamada": _ms(t_llamada0, t_llamada1),
            "bloques": bloques,
        }
        return salida, tiempos

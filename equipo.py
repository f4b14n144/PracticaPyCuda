import ctypes
import os
import platform
import subprocess
import sys

from filtros.comun import RAIZ


def comando(*args):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def cpu():
    nombre = platform.processor()
    nucleos = ""
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                nombre = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
        nucleos = comando("powershell", "-NoProfile", "-Command",
                          "(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum")
    return f"{nombre}, {nucleos or '?'} nucleos, {os.cpu_count()} hilos logicos"


def ram_gb():
    if sys.platform == "win32":
        class Estado(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
        e = Estado()
        e.dwLength = ctypes.sizeof(Estado)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(e))
        return e.ullTotalPhys / 1024 ** 3
    with open("/proc/meminfo") as f:
        return int(f.readline().split()[1]) / 1024 ** 2


def main():
    import numpy
    import PIL
    lineas = [
        "Equipo y herramientas usados en crudo_python.csv, crudo_c.csv y validacion.csv",
        f"CPU: {cpu()}",
        f"RAM: {ram_gb():.1f} GB visibles para el sistema",
        f"Sistema: {platform.platform()}",
        f"Python {platform.python_version()}, NumPy {numpy.__version__}, Pillow {PIL.__version__}",
    ]
    try:
        import pycuda
        import pycuda.driver as cuda
        cuda.init()
        d = cuda.Device(0)
        drv = cuda.get_driver_version()
        lineas += [
            f"PyCUDA {pycuda.VERSION_TEXT}",
            f"GPU: {d.name()}, {d.total_memory() // 1024 ** 2} MiB, capacidad {d.compute_capability()}",
            f"API del controlador CUDA: {drv // 1000}.{(drv % 1000) // 10}",
        ]
    except Exception as e:  # noqa: BLE001 - se registra lo que haya
        lineas.append(f"PyCUDA no disponible: {e}")
    smi = comando("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")
    lineas.append(f"Controlador NVIDIA: {smi or '?'}")
    nvcc = [l for l in comando("nvcc", "--version").splitlines() if "release" in l]
    lineas.append(f"CUDA Toolkit: {nvcc[0].strip() if nvcc else '?'}")
    for paquete in ("pandas", "matplotlib"):
        try:
            lineas.append(f"{paquete} {__import__(paquete).__version__}")
        except ImportError:
            pass
    lineas.append("Alimentacion durante las mediciones: (completar: corriente / bateria)")
    ruta = RAIZ / "experimentos" / "equipo.txt"
    ruta.parent.mkdir(exist_ok=True)
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("\n".join(lineas))
    print(f"\nGuardado en {ruta}")


if __name__ == "__main__":
    main()

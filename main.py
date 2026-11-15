import sys
import time

from pathlib import Path

from filtros.constantes import LAPLACIANO, MEDIA, NITIDEZ, NOMBRES

RAIZ = Path(__file__).resolve().parent


def pedir_entero(mensaje, validos=None):
    try:
        valor = int(input(mensaje))
    except ValueError:
        return None
    return valor if validos is None or valor in validos else None


def main():
    modo = pedir_entero("Seleccione modo de ejecucion:\n1) Python secuencial\n2) PyCUDA (GPU)\n", (1, 2))
    if modo is None:
        print("Modo invalido")
        return 1
    filtro = pedir_entero("Seleccione filtro:\n1) Media\n2) Nitidez\n3) Laplaciano\n", (MEDIA, NITIDEZ, LAPLACIANO))
    if filtro is None:
        print("Filtro invalido")
        return 1
    opcion = pedir_entero("Seleccione resolucion:\n1) 2000x2000\n2) 4000x4000\n3) 6000x6000\n", (1, 2, 3))
    if opcion is None:
        print("Resolucion invalida")
        return 1
    resolucion = {1: 2000, 2: 4000, 3: 6000}[opcion]

    mascara = 3
    if filtro == MEDIA:
        mascara = pedir_entero("Ingrese tamano de mascara impar (ejemplo: 3, 9, 21): ")
        if mascara is None or mascara < 3 or mascara % 2 == 0:
            print("Tamano de mascara invalido")
            return 1

    hilos = 256
    if modo == 2:
        hilos = pedir_entero("Ingrese hilos por bloque CUDA (128, 256, 512): ", (128, 256, 512))
        if hilos is None:
            print("Cantidad de hilos invalida")
            return 1

    ruta_entrada = RAIZ / "Imagenes" / "entrada" / f"{resolucion}.png"
    ruta_salida = RAIZ / "Imagenes" / "final" / "imagen_final.png"
    ruta_salida.parent.mkdir(parents=True, exist_ok=True)

    if modo == 1:
        # Python secuencial: sin NumPy, pixel por pixel.
        from filtros import secuencial
        entrada, ancho, alto = secuencial.cargar_gris(ruta_entrada)
        print(f"Imagen {ancho}x{alto}, {NOMBRES[filtro]} k={mascara}. Ejecutando filtro...")
        t0 = time.perf_counter()
        salida = secuencial.aplicar_filtro(entrada, ancho, alto, filtro, mascara)
        print(f"Tiempo secuencial: {(time.perf_counter() - t0) * 1000:.3f} ms")
        secuencial.guardar_gris(ruta_salida, secuencial.convertir_a_visualizacion(salida, filtro), ancho, alto)
    else:
        from filtros.comun import cargar_gris, convertir_a_visualizacion, guardar_gris
        from filtros.gpu import FiltrosGPU
        entrada = cargar_gris(ruta_entrada)
        alto, ancho = entrada.shape
        print(f"Imagen {ancho}x{alto}, {NOMBRES[filtro]} k={mascara}. Ejecutando filtro...")
        with FiltrosGPU() as gpu:
            print(f"GPU:          {gpu.nombre}")
            print(f"Contexto:     {gpu.t_contexto_ms:.3f} ms")
            print(f"Compilacion:  {gpu.t_compilacion_ms:.3f} ms")
            salida, t = gpu.ejecutar(entrada, filtro, mascara, hilos)
        print(f"Bloques:      {t['bloques']} de {hilos} hilos")
        print(f"CPU->GPU:     {t['t_cpu_a_gpu']:.3f} ms")
        print(f"Kernel:       {t['t_kernel']:.3f} ms")
        print(f"GPU->CPU:     {t['t_gpu_a_cpu']:.3f} ms")
        print(f"Conjunto:     {t['t_conjunto']:.3f} ms")
        print(f"Con reservas: {t['t_llamada']:.3f} ms")
        guardar_gris(ruta_salida, convertir_a_visualizacion(salida, filtro))

    print(f"Imagen guardada en: {ruta_salida}\nProceso completado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

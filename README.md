# Convolución de imágenes de alta resolución con PyCUDA

Práctica 02 de Computación Paralela. Continúa la [práctica 01](https://github.com/Jellz01/PracticaCUDAvsCPU) (commit `f82b51b409cb12cd69382ef3c182176dbdd12864`): el programa anfitrión pasa a Python y los kernels de CUDA C se ejecutan con PyCUDA. Se comparan cuatro versiones: C/C++ secuencial, CUDA C, Python secuencial y PyCUDA.

## Qué se migró y qué se reutilizó

| Parte | Práctica 01 (C/C++) | Práctica 02 (Python) |
|---|---|---|
| Carga y escala de grises | `stb_image` | Pillow → `bytes` (secuencial) o `numpy.uint8` contiguo (PyCUDA) |
| Máscaras y selección de filtro | `filtros.h`, `main.cpp` | `filtros/comun.py`, `main.py` |
| Convolución secuencial | `filtros.cpp` | `filtros/secuencial.py`: bucles explícitos, Python puro, sin NumPy |
| Gestión de memoria, transferencias, lanzamiento y tiempos | `filtros.cu` (`ejecutarCUDA`) | `filtros/gpu.py` (`FiltrosGPU.ejecutar`) |
| Cálculo por píxel en GPU | `filtros.h` | **El mismo archivo** (`filtros/kernels/filtros.h`, copia sin cambios), incluido con `#include` |
| Kernels | `filtros.cu` | `filtros/kernels/filtros_pycuda.cu`: mismo cuerpo, solo se agrega `extern "C"` |
| Validación | `pruebas.cpp` | `pruebas.py` |
| Experimentos | `benchmark.cpp` | `benchmark.py` |
| Conversión a 0-255 y guardado | `filtros.cpp` + `stb_image_write` | `convertir_a_visualizacion()` + Pillow |

`extern "C"` es necesario porque PyCUDA obtiene cada kernel por su nombre con `get_function()`, y sin él el compilador de C++ cambia el nombre (*name mangling*).

## Estructura

| Ruta | Contenido |
|---|---|
| `main.py` | Programa interactivo: Python secuencial o PyCUDA |
| `filtros/comun.py` | Tipos, máscaras, carga, conversión a 0-255, errores, SHA-256 |
| `filtros/secuencial.py` | Convolución secuencial en Python puro, sin NumPy (entrada `bytes`, salida `array('f')`) |
| `filtros/constantes.py` | Filtros y máscaras, sin dependencias |
| `filtros/gpu.py` | Programa anfitrión PyCUDA: contexto, compilación, memoria, transferencias, lanzamiento y eventos |
| `filtros/kernels/filtros_pycuda.cu` | Kernels propios (los de la práctica 01) |
| `filtros/kernels/filtros.h` | Copia sin cambios de `codigo/filtros.h` de la práctica 01 |
| `pruebas.py` | Prueba manual, límites, comparación de las cuatro versiones, evidencia visual |
| `benchmark.py` | Experimentos de Python secuencial y PyCUDA |
| `equipo.py` | Registra hardware y versiones en `experimentos/equipo.txt` |
| `experimentos/graficar.py` | Resumen, aceleraciones y gráficas de las cuatro versiones |
| `Imagenes/entrada/` | Las mismas entradas de la práctica 01, con `SHA256SUMS.txt` |
| `experimentos/crudo_p01.csv` | Tiempos de C/C++ y CUDA C, medidos con `benchmark.exe` de la práctica 01 |
| `referencias/` | Salidas de C/C++ y CUDA C para la validación (no se versiona, ver abajo) |
| `ejecutar_todo.ps1` | Ejecuta toda la secuencia |

### Relación con el repositorio de la práctica 01

El código C/C++ y CUDA C está en [PracticaCUDAvsCPU](https://github.com/Jellz01/PracticaCUDAvsCPU):

- Commit `f82b51b409cb12cd69382ef3c182176dbdd12864`: versión de la práctica 01 (código C/C++ y CUDA C sin cambios) cuyo `benchmark.exe` genera `crudo_p01.csv`.
- Commit `1a721f59d8d0ee83d39a62f8551af53b46bc7c03`: agrega `codigo/exportar_salidas.cpp`, que guarda las salidas `float` de C/C++ y CUDA C en archivos `.f32` para compararlas desde Python. No modifica `filtros.cpp`, `filtros.cu` ni `filtros.h`.

## Entorno

Requisitos en Windows:

- GPU NVIDIA, controlador y CUDA Toolkit (`nvcc` en el `PATH`).
- Visual Studio Build Tools con C++ (`cl.exe`). PyCUDA compila los kernels con `nvcc`, que en Windows necesita `cl.exe`. Si no está en el `PATH`, `filtros/gpu.py` lo busca con `vswhere` y lo agrega al proceso, así que también funciona desde PyCharm.
- Python 3 con las dependencias de `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
pip freeze > requirements-lock.txt     # versiones exactas usadas en las mediciones
python equipo.py                       # escribe experimentos/equipo.txt
```

En Windows `pip` compila PyCUDA desde el código fuente. Si falla, abrir la terminal "x64 Native Tools Command Prompt" de Visual Studio, comprobar `nvcc --version` y `cl`, y repetir `pip install pycuda`. Si la versión de Python es demasiado nueva para PyCUDA, crear el entorno con Python 3.12.

## Ejecución

Desde la raíz del proyecto:

Primero, en el repositorio de la práctica 01, con el entorno de Visual Studio cargado:

```powershell
cd codigo
cmake -S . -B build-local -G "NMake Makefiles"
cmake --build build-local
.\exportar_salidas.exe ..\Imagenes\entrada <ruta de este proyecto>\referencias
.\benchmark.exe ..\Imagenes\entrada <ruta de este proyecto>\experimentos\crudo_p01.csv 10 2
```

Luego, desde la raíz de este proyecto:

```powershell
python main.py                         # programa interactivo
python pruebas.py                      # validación; escribe experimentos/validacion.csv
python benchmark.py --solo-estimar     # prueba pequeña de viabilidad de Python secuencial
python benchmark.py                    # experimentos; escribe experimentos/crudo_python.csv
python experimentos\graficar.py        # resumen.csv, aceleraciones.csv, cuda_vs_pycuda.csv, gráficas
```

`ejecutar_todo.ps1` hace toda la secuencia. Si el benchmark se interrumpe, `python benchmark.py --reanudar` continúa sin repetir lo terminado.

## Distribución del trabajo en la GPU

Igual que en la práctica 01: bloques e hilos solo en X y un píxel de salida por hilo.

```text
bloques = ceil(ancho × alto / hilosPorBloque)        # se calcula en Python
indice  = blockIdx.x × blockDim.x + threadIdx.x      # en el kernel
fila    = indice / ancho
columna = indice % ancho
```

Desde Python el lanzamiento es `kernel.prepared_call((bloques, 1), (hilos, 1, 1), ...)`. Los hilos con `indice >= ancho × alto` (los sobrantes del último bloque) terminan sin hacer nada. Los vecinos se leen de memoria global y la entrada y la salida son arreglos distintos.

## Tipos

| Dato | NumPy | Kernel / C | Firma en `prepare()` |
|---|---|---|---|
| Imagen de entrada | `uint8`, contiguo, fila mayor | `const unsigned char*` | `P` |
| Salida | `float32`, contiguo | `float*` | `P` |
| Ancho, alto, máscara | `int32` | `int` | `i` |

`ejecutar()` comprueba que la entrada sea `uint8` contigua (`flags.c_contiguous`), y la salida se crea con `np.empty(..., float32)`, que siempre es contigua.

## Equivalencia entre las cuatro versiones

- Extensión con ceros: un vecino fuera de la imagen vale 0.
- Acumulación entera: en C es `int` (máximo 441 × 255 = 112455, no desborda) y en Python un entero exacto.
- Salida `float32` sin convertir (en Python secuencial, `array('f')` de la biblioteca estándar, que es el `float` de C). Nitidez y laplaciano: `float32(suma)` es exacto. Media: Python divide en `float64` y guarda en `float32`; como 53 ≥ 2·24 + 2, ese doble redondeo da exactamente el mismo valor que la división `float` de C. En las pruebas, Python secuencial coincide bit a bit con C/C++.
- La conversión a 0-255 (redondeo, saturación, +128 en laplaciano) se aplica solo para guardar imágenes. Los errores se miden antes.

## Validación (`pruebas.py`)

- **Entradas:** comprueba el SHA-256 de `2000.png`, `4000.png` y `6000.png` contra `SHA256SUMS.txt` de la práctica 01.
- **Prueba manual:** matriz 3×3 `[10 20 30; 40 50 60; 70 80 90]` con los resultados calculados a mano (README de la práctica 01), para las cuatro versiones y 128, 256 y 512 hilos.
- **Límites:** 1×1, 7×5, 513×1, 1×257 y 1001×777 (ningún total es múltiplo de 128, 256 ni 512), con los mismos datos pseudoaleatorios que `pruebas.cpp`. Antes del kernel la salida en la GPU se llena con `NaN`; si algún píxel queda sin escribir, la prueba falla (`pixeles_sin_escribir`).
- **Comparación numérica:** referencia C/C++ secuencial. Error absoluto máximo y medio de Python secuencial, CUDA C y PyCUDA, antes de convertir a 0-255. Python secuencial se valida en 2000×2000 (y en 4000×4000 3×3 con `--completo`) por su tiempo de ejecución.
- **Tolerancia: 1e-4.** Es la de la práctica 01. Cerca de 255, dos `float32` consecutivos se separan 1.5e-5, así que 1e-4 admite solo unos pocos pasos de redondeo y queda muy por debajo de 0.5, la diferencia que cambiaría un nivel de gris.
- **Evidencia visual:** `experimentos/imagenes/` con la original, media 3, media 21, nitidez y laplaciano en 2000×2000, y `recortes_2000.png` con la misma región de 300×300 en todas.

## Metodología de medición

- **Configuraciones:** las de la guía. Fuera del experimento de hilos se usan 256 hilos por bloque. Las mediciones repetidas se comparten entre experimentos (columna `experimentos`).
- **Repeticiones:** PyCUDA 2 de calentamiento y 10 medidas; Python secuencial 1 de calentamiento sobre un recorte de 200×200 (en Python puro solo prepara el intérprete) y 5 medidas sobre la imagen completa. El calentamiento queda en el CSV con `fase = calentamiento` y no entra en el resumen.
- **Python secuencial:** `time.perf_counter()` solo alrededor de la convolución.
- **PyCUDA:** CPU→GPU, GPU→CPU y conjunto con `perf_counter` y `Context.synchronize()`; kernel con `pycuda.driver.Event`, sincronizando antes de `time_since()`.
- **Reserva de memoria:** fuera del tiempo conjunto, como en CUDA C. El arreglo de salida en la CPU se reserva una vez por configuración y se reutiliza entre repeticiones, igual que el `std::vector` de `benchmark.cpp`. `t_llamada` registra además la llamada completa desde Python, con reserva y liberación.
- **Preparación:** el contexto y la compilación del kernel se miden una vez por ejecución del programa en `experimentos/preparacion.csv`, sin caché de compilación.
- **Excluido:** lectura y escritura de imágenes y conversión a 0-255.
- **Viabilidad de Python secuencial:** `benchmark.py` estima primero el tiempo con un recorte de 200×200 (solo informativo, nunca se registra como medición). Cada ejecución se corta al pasar `--limite-min` (30 min por defecto); entonces se registra `estado = no_completada` con el tiempo transcurrido y las filas procesadas. Las configuraciones incompletas no entran en el resumen ni en las aceleraciones.
- **Aceleraciones:** CUDA C respecto a C/C++ secuencial y PyCUDA respecto a Python secuencial, con el kernel solo y con transferencias. `cuda_vs_pycuda.csv` compara CUDA C y PyCUDA en las mismas configuraciones.

## Resultados

| Archivo | Contenido |
|---|---|
| `experimentos/crudo_p01.csv` | Registros de C/C++ y CUDA C (código de la práctica 01, medidos de nuevo) |
| `experimentos/crudo_python.csv` | Registros de Python secuencial y PyCUDA |
| `experimentos/preparacion.csv` | Contexto y compilación de PyCUDA |
| `experimentos/resumen.csv` | Mediana, cuartiles, IQR, desviación y CV por configuración |
| `experimentos/aceleraciones.csv` | Aceleraciones de cómputo y con transferencias |
| `experimentos/cuda_vs_pycuda.csv` | Razón PyCUDA / CUDA C por componente |
| `experimentos/validacion.csv` | Errores de las cuatro versiones |
| `experimentos/equipo.txt` | Hardware y versiones |

Todos los tiempos son nuevos. Los de C/C++ y CUDA C se volvieron a medir con `benchmark.exe` del commit `f82b51b`, sin cambios, porque el controlador NVIDIA cambió entre las dos prácticas (617.14 en la práctica 01, 621.14 ahora). De la práctica 01 solo se reutilizan el código C/C++ y CUDA C y las imágenes de entrada (verificadas con SHA-256).

## Datos de entrada

Las mismas imágenes de la práctica 01: `ResizeImage.py` de ese repositorio parte de `8k image.jpg` ([Boat at Balos Beach](https://unsplash.com/photos/white-boat-at-balos-beach-sKOBxEb6E-E), Joshua Kettle, licencia de Unsplash), recorta 6000×6000, convierte a gris y reduce a 4000 y 2000 con LANCZOS.

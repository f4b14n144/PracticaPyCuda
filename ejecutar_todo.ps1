# Reproduce la practica 02. Ejecutar desde la raiz del proyecto con el entorno virtual
# activo. Antes, en el repositorio de la practica 01 (con el entorno de Visual Studio):
#   cd codigo; cmake -S . -B build-local -G "NMake Makefiles"; cmake --build build-local
#   .\exportar_salidas.exe ..\Imagenes\entrada <ruta de este proyecto>\referencias
$ErrorActionPreference = "Stop"

python equipo.py
python pruebas.py | Tee-Object -FilePath experimentos\pruebas.log
python benchmark.py --reanudar
python experimentos\graficar.py

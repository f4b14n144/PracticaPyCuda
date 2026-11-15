#ifndef FILTROS_H
#define FILTROS_H

// Las funciones HD se compilan para CPU y para GPU, asi ambas versiones usan
// exactamente las mismas mascaras, los mismos tipos y la misma extension de bordes.
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

enum TipoFiltro {
    FILTRO_MEDIA = 1,
    FILTRO_NITIDEZ = 2,
    FILTRO_LAPLACIANO = 3
};

// Extension con ceros: todo pixel fuera de la imagen vale 0.
HD inline int pixelConCeros(const unsigned char* entrada, int ancho, int alto, int x, int y) {
    if (x < 0 || x >= ancho || y < 0 || y >= alto) return 0;
    return entrada[y * ancho + x];
}

// Media k x k: todos los coeficientes valen 1/(k*k).
HD inline float calcularMedia(const unsigned char* entrada, int ancho, int alto, int x, int y, int ksize) {
    int radio = ksize / 2;
    int suma = 0;
    for (int ky = -radio; ky <= radio; ky++)
        for (int kx = -radio; kx <= radio; kx++)
            suma += pixelConCeros(entrada, ancho, alto, x + kx, y + ky);
    return (float)suma / (float)(ksize * ksize);
}

// Nitidez 3 x 3: [0, -1, 0; -1, 5, -1; 0, -1, 0]
HD inline float calcularNitidez(const unsigned char* entrada, int ancho, int alto, int x, int y) {
    const int mascara[3][3] = {{0, -1, 0}, {-1, 5, -1}, {0, -1, 0}};
    int suma = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++)
            suma += mascara[ky + 1][kx + 1] * pixelConCeros(entrada, ancho, alto, x + kx, y + ky);
    return (float)suma;
}

// Laplaciano 3 x 3: [0, 1, 0; 1, -4, 1; 0, 1, 0]
HD inline float calcularLaplaciano(const unsigned char* entrada, int ancho, int alto, int x, int y) {
    const int mascara[3][3] = {{0, 1, 0}, {1, -4, 1}, {0, 1, 0}};
    int suma = 0;
    for (int ky = -1; ky <= 1; ky++)
        for (int kx = -1; kx <= 1; kx++)
            suma += mascara[ky + 1][kx + 1] * pixelConCeros(entrada, ancho, alto, x + kx, y + ky);
    return (float)suma;
}

// Tiempos de una ejecucion CUDA, en milisegundos.
// La reserva de memoria se hace antes de medir y no esta incluida.
struct TiemposCUDA {
    double h2d_ms;     // transferencia CPU -> GPU
    double kernel_ms;  // ejecucion del kernel, medida con eventos CUDA
    double d2h_ms;     // transferencia GPU -> CPU
    double total_ms;   // tiempo conjunto de transferencias y ejecucion
    int bloques;       // bloques usados para cubrir la imagen
};

// Version secuencial: la salida es numerica, sin convertir a 0-255.
void aplicarFiltroMedia(const unsigned char* entrada, float* salida, int ancho, int alto, int ksize);
void aplicarFiltroNitidez(const unsigned char* entrada, float* salida, int ancho, int alto);
void aplicarFiltroLaplaciano(const unsigned char* entrada, float* salida, int ancho, int alto);

// Conversion al rango de visualizacion, solo despues de validar los valores:
//   Media:      redondeo al entero mas cercano.
//   Nitidez:    saturacion a [0, 255].
//   Laplaciano: se suma 128 para ver los negativos y se satura a [0, 255].
unsigned char convertirAVisualizacion(float valor, TipoFiltro filtro);

// Version CUDA: reparte el trabajo en bloques e hilos solo en X.
void filtroMediaCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto, int ksize,
                     int hilosPorBloque, TiemposCUDA* tiempos);
void filtroNitidezCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto,
                       int hilosPorBloque, TiemposCUDA* tiempos);
void filtroLaplacianoCUDA(const unsigned char* h_entrada, float* h_salida, int ancho, int alto,
                          int hilosPorBloque, TiemposCUDA* tiempos);

// Maximo de hilos por bloque que permite la GPU para estos kernels.
int maxHilosPorBloqueCUDA();

#endif // FILTROS_H

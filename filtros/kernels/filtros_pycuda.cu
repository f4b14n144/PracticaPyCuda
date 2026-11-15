#include "filtros.h"

extern "C" __global__ void kernelFiltroMedia(const unsigned char* entrada, float* salida,
                                             int ancho, int alto, int ksize) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularMedia(entrada, ancho, alto, x, y, ksize);
}

extern "C" __global__ void kernelFiltroNitidez(const unsigned char* entrada, float* salida,
                                               int ancho, int alto) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularNitidez(entrada, ancho, alto, x, y);
}

extern "C" __global__ void kernelFiltroLaplaciano(const unsigned char* entrada, float* salida,
                                                  int ancho, int alto) {
    int indice = blockIdx.x * blockDim.x + threadIdx.x;
    if (indice >= ancho * alto) return;
    int y = indice / ancho;
    int x = indice % ancho;
    salida[indice] = calcularLaplaciano(entrada, ancho, alto, x, y);
}

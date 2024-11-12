#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Dimensions des matrices
#define N 3  // Taille des matrices (N x N)

// Kernel CUDA pour la multiplication de matrices
__global__ void matrixMultiply(float* A, float* B, float* C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int size = N * N * sizeof(float);

    // Allouer la mémoire sur le CPU (hôte)
    float* h_A = (float*)malloc(size);
    float* h_B = (float*)malloc(size);
    float* h_C = (float*)malloc(size);

    // Initialiser les matrices A et B avec des valeurs
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    // Allouer la mémoire sur le GPU (périphérique)
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copier les matrices A et B du CPU vers le GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Définir la taille des blocs et des grilles
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // Lancer le kernel pour la multiplication de matrices
    matrixMultiply << <grid, block >> > (d_A, d_B, d_C);

    // Attendre la fin de l'exécution du kernel
    cudaDeviceSynchronize();

    // Copier le résultat du GPU vers le CPU
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Afficher un élément du résultat pour vérifier
    for (int i = 0; i < N * N; i++) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    // Libérer la mémoire
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
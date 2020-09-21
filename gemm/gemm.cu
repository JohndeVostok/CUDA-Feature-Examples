#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>
#include "cublas_v2.h"
#include <chrono>

void gemmCPU(double *A, double *B, double *C, int m, int n, int k, int a, int b) {
    for (int i = 0; i < m; i++) {
        for (int t = 0; t < k; t++) {
            for (int j = 0; j < n; j++) {
                C[i * n + j] += A[i * k + t] * B[t * n + j];
            }
        }
    }
}

void print_matrix(double *A, int m, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%f ", A[i * n + j]);
        }
        printf("\n");
    }
}

int main() {
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    // cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);

//    int m = atoi(getenv("CUOPT_GEMM_M"));
//    int n = atoi(getenv("CUOPT_GEMM_N"));
//    int k = atoi(getenv("CUOPT_GEMM_K"));
    int m = 4096, n = 4096, k = 4096;

    printf("%d %d %d\n", m, n, k);

    uint32_t rowsA = m, colsA = k, rowsB = k, colsB = n, rowsC = m, colsC = n;
    size_t matrixSizeA = (size_t)rowsA * colsA, matrixSizeB = (size_t)rowsB * colsB, matrixSizeC = (size_t)rowsC * colsC;
    double *devPtrA = 0, *devPtrB = 0, *devPtrC = 0, *devPtrD = 0;
    double a = 1, b = 1;
    
    cudaMalloc((void**)&devPtrA, matrixSizeA * sizeof(double));
    cudaMalloc((void**)&devPtrB, matrixSizeB * sizeof(double));
    cudaMalloc((void**)&devPtrC, matrixSizeC * sizeof(double));
    cudaMalloc((void**)&devPtrD, matrixSizeC * sizeof(double));
    double *A = (double *)malloc(matrixSizeA * sizeof(double));
    double *B = (double *)malloc(matrixSizeB * sizeof(double));
    double *C = (double *)malloc(matrixSizeC * sizeof(double));
    double *C_cpu = (double *)malloc(matrixSizeC * sizeof(double));
    double *C_gpu = (double *)malloc(matrixSizeC * sizeof(double));
    double *C_tc = (double *)malloc(matrixSizeC * sizeof(double));
    for (int i = 0; i < matrixSizeA; i++) A[i] = rand() % 5;
    for (int i = 0; i < matrixSizeB; i++) B[i] = rand() % 5;
    for (int i = 0; i < matrixSizeC; i++) C[i] = rand() % 5;
    for (int i = 0; i < matrixSizeC; i++) C_cpu[i] = C[i];
    for (int i = 0; i < matrixSizeC; i++) C_gpu[i] = C[i];
    for (int i = 0; i < matrixSizeC; i++) C_tc[i] = C[i];
    
    cublasSetMatrix(rowsA, colsA, sizeof(double), A, rowsA, devPtrA, rowsA);
    cublasSetMatrix(rowsB, colsB, sizeof(double), B, rowsB, devPtrB, rowsB);
    cublasSetMatrix(rowsC, colsC, sizeof(double), C, rowsC, devPtrC, rowsC);
    cublasSetMatrix(rowsC, colsC, sizeof(double), C, rowsC, devPtrD, rowsC);

    // gemmCPU(A, B, C_cpu, m, n, k, a, b);
    // uint64_t time_tc = 0;
    for (int i = 0; i < 13; i++) {
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, devPtrA, k, devPtrB, k, &b, devPtrC, m);
    }
    cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    for (int i = 0; i < 13; i++) {
        cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, k, &a, devPtrA, k, devPtrB, k, &b, devPtrC, m);
    }
    // printf("TC: %f\n", time_tc / 256.0);

    // uint64_t time_gpu = 0;
    // for (int i = 0; i < 1024; i++) {
    //     auto start = std::chrono::system_clock::now();
    //     cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &a, devPtrB, n, devPtrA, k, &b, devPtrC, n);
    //     auto end = std::chrono::system_clock::now();
    //     time_gpu += (uint64_t)std::chrono::duration_cast<std::chrono::microseconds>(end-start).count();
    // }
    // printf("GPU: %lu\n", time_gpu);

    // cublasGetMatrix(rowsC, colsC, sizeof(double), devPtrC, rowsC, C_gpu, rowsC);
    // cublasGetMatrix(rowsC, colsC, sizeof(double), devPtrD, rowsC, C_tc, rowsC);

    // bool flag = true;
    // for (int i = 0; i < m; i++) {
    //     for (int j = 0; j < n; j++) {
    //         if (C_tc[i * n + j] != C_gpu[i * n + j]) {
    //             flag = false;
    //         }
    //     }
    // }
    // if (flag) printf("Validated.\n"); else printf("Unvalidated.\n");

    cudaFree(devPtrA);
    cudaFree(devPtrB);
    cudaFree(devPtrC);
    free(A);
    free(B);
    free(C);

    cublasDestroy(handle);
    return 0;
}

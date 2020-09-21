// Shared memory unexpected bank-conflict.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include <cuda.h>
#include <mma.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

using namespace nvcuda;

__global__ void latency_hiding(float *data, int size, uint64_t *clock) {
    extern __shared__ int4 tmp[];
    uint64_t clock_start = clock64();

    int4 dest;
    dest = tmp[threadIdx.x];
    dest.x += 1;
    dest.y += 1;
    dest.z += 1;
    dest.w += 1;
    tmp[threadIdx.x] = dest;

    uint64_t clock_end = clock64();
    // atomicAdd(reinterpret_cast<unsigned long long *>(clock), clock_end - clock_start);
}

int main(int argc, char *argv[]) {
	cudaError_t cuda_status;
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		printf("cudaSetDevice failed! ");
		return 1;
	}

    half *devPtrA = 0, *devPtrB = 0;
    float *devPtrC = 0, *devPtrD = 0;
    float alpha = 1, beta = 1;
    uint32_t *devPtrDebug = 0;

    cudaMalloc((void **) &devPtrA, 65536 * sizeof(half));
    cudaMalloc((void **) &devPtrB, 65536 * sizeof(half));
    cudaMalloc((void **) &devPtrC, 65536 * sizeof(float));
    cudaMalloc((void **) &devPtrD, 65536 * sizeof(float));
    cudaMalloc((void **) &devPtrDebug, 65536);
	
    half *ptrA = (half *) malloc(65536 * sizeof(half));
    half *ptrB = (half *) malloc(65536 * sizeof(half));
    float *ptrC = (float *) malloc(65536 * sizeof(float));
    float *ptrD = (float *) malloc(65536 * sizeof(float));
    float *ptrE = (float *) malloc(65536 * sizeof(float));
    uint32_t *ptrDebug = (uint32_t *) malloc(65536);

    memset(ptrC, 0, 65536 * sizeof(float));
    cudaMemcpy(devPtrA, ptrA, 65536 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrB, ptrB, 65536 * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(devPtrC, ptrC, 65536 * sizeof(float), cudaMemcpyHostToDevice);

    uint64_t *clk;

    cudaMallocManaged(&clk, sizeof(uint64_t));

    cudaError_t result;
    result = cudaFuncSetAttribute(latency_hiding, cudaFuncAttributeMaxDynamicSharedMemorySize, 72000);
    if (result != cudaSuccess) {
        return 0;
    }
    result = cudaFuncSetAttribute(latency_hiding, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    if (result != cudaSuccess) {
        return 0;
    }
    latency_hiding<<<160, 1024, 72000>>>(devPtrC, 64, clk);
    cudaDeviceSynchronize();
    printf("%lu\n", *clk);

    cuda_status = cudaDeviceReset();
	if (cuda_status != cudaSuccess) {
		printf("cudaDeviceReset failed! ");
		return 1;
	}

	return 0;
}
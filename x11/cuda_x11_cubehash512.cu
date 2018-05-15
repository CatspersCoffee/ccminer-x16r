/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
	Optimized for nvidia pascal by sp (2018)
*/

#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"
#include "cuda_x11_aes_alexis.cuh"
#include "cubehash/cubehash512.cuh"
#include "shavite/shavite512.cuh"

#define TPB 1024
#define CUBEHASH_SHAVITE_TPB 128

/***************************************************/
// GPU Hash Function
__global__
void x11_cubehash512_gpu_hash_64(uint32_t threads, uint64_t *g_hash){

	uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads){
		x11_cubehash512_gpu_hash_64_unroll_10r((uint32_t*)&g_hash[8 * thread]);
	}
}

__global__
//__launch_bounds__(384, 2)
__launch_bounds__(CUBEHASH_SHAVITE_TPB, 3)
void x11_cubehashShavite512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);
    __shared__ uint32_t sharedMemory[4][256];
    aes_gpu_init128(sharedMemory);

    uint32_t *const hash = &g_hash[thread << 4];

    x11_cubehash512_gpu_hash_64(hash);
    __syncthreads();
    shavite512(sharedMemory, hash);
}


__host__
void x11_cubehash512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash){

    // berechne wie viele Thread Blocks wir brauchen
    dim3 grid((threads + TPB-1)/TPB);
    dim3 block(TPB);

    x11_cubehash512_gpu_hash_64<<<grid, block>>>(threads, (uint64_t*)d_hash);
}

__host__
void x11_cubehash_shavite512_cpu_hash_64(int thr_id, uint32_t threads, uint32_t *d_hash)
{

	dim3 grid((threads + CUBEHASH_SHAVITE_TPB - 1) / CUBEHASH_SHAVITE_TPB);
	dim3 block(CUBEHASH_SHAVITE_TPB);

	x11_cubehashShavite512_gpu_hash_64 << <grid, block >> > (threads, d_hash);
}

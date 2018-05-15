/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
*/
#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"

#define INTENSIVE_GMF
#include "cuda_x11_aes_alexis.cuh"
#include "shavite/shavite512.cuh"

#define TPB 128

// GPU Hash
//__global__ __launch_bounds__(TPB,8) /* 5820 */
__global__ __launch_bounds__(TPB,7) /* 5900 */
//__global__ __launch_bounds__(TPB, 6) /* 5775 */
void x11_shavite512_gpu_hash_64_alexis(const uint32_t threads, uint32_t *g_hash) {
    __shared__ uint32_t sharedMemory[4][256];
	aes_gpu_init128(sharedMemory);

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	if (thread < threads) {
      uint32_t *hash = &g_hash[thread<<4];
      __syncthreads();
      shavite512(sharedMemory, hash);
    }
}

__host__
void x11_shavite512_cpu_hash_64_alexis(int thr_id, uint32_t threads, uint32_t *d_hash)
{
	dim3 grid((threads + TPB-1)/TPB);
	dim3 block(TPB);

	// note: 128 threads minimum are required to init the shared memory array
	x11_shavite512_gpu_hash_64_alexis<<<grid, block>>>(threads, d_hash);
}

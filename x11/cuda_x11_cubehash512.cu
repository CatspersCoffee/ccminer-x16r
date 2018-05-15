/*
	Based on Tanguy Pruvot's repo
	Provos Alexis - 2016
	Optimized for nvidia pascal by sp (2018)
*/

#include "cuda_helper_alexis.h"
#include "cuda_vectors_alexis.h"
#include "cuda_x11_aes_alexis.cuh"
#include "cubehash/cubehash.h"

//--SHAVITE--------------------------------------------
__device__
static void round_3_7_11(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	x = p[2] ^ *(uint4*)&r[0];
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	r[4] ^= r[0];
	r[5] ^= r[1];
	r[6] ^= r[2];
	r[7] ^= r[3];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x = x ^ *(uint4*)&r[4];
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	r[8] ^= r[4];
	r[9] ^= r[5];
	r[10] ^= r[6];
	r[11] ^= r[7];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[8];
	x.y ^= r[9];
	x.z ^= r[10];
	x.w ^= r[11];
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	r[12] ^= r[8];
	r[13] ^= r[9];
	r[14] ^= r[10];
	r[15] ^= r[11];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x.x ^= r[12];
	x.y ^= r[13];
	x.z ^= r[14];
	x.w ^= r[15];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[1].x ^= x.x;
	p[1].y ^= x.y;
	p[1].z ^= x.z;
	p[1].w ^= x.w;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	x = p[0] ^ *(uint4*)&r[16];
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[20];
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[24];
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[3] ^= x;
}

__device__
static void round_4_8_12(const uint32_t sharedMemory[4][256], uint32_t* r, uint4 *p, uint4 &x){
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	x = p[1] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &x);

	r[4] ^= r[29];	r[5] ^= r[30];
	r[6] ^= r[31];	r[7] ^= r[0];

	x ^= *(uint4*)&r[4];
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[8];
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[0] ^= x;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	x = p[3] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	x ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &x);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	x ^= *(uint4*)&r[24];
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	AES_ROUND_NOKEY(sharedMemory, &x);
	x ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &x);
	p[2] ^= x;
}

//--END OF SHAVITE MACROS------------------------------------


#define TPB 1024

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
__launch_bounds__(128, 5)
void x11_cubehashShavite512_gpu_hash_64(uint32_t threads, uint32_t *g_hash){

	const uint32_t thread = (blockDim.x * blockIdx.x + threadIdx.x);

	__shared__ uint32_t sharedMemory[4][256];

	aes_gpu_init128(sharedMemory);
	//	if (thread < threads)
	//	{
	uint32_t *const hash = &g_hash[thread * 16U];

	//Cubehash

	x11_cubehash512_gpu_hash_64(hash);

	uint4 y;
	uint32_t r[32];
	uint4 msg[4];
	// kopiere init-state
	uint4 p[4];
	const uint32_t state[16] = {
		0x72FCCDD8, 0x79CA4727, 0x128A077B, 0x40D55AEC, 0xD1901A06, 0x430AE307, 0xB29F5CD1, 0xDF07FBFC,
		0x8E45D73D, 0x681AB538, 0xBDE86578, 0xDD577E47, 0xE275EADE, 0x502D9FCD, 0xB9357178, 0x022A4B9A
	};
	*(uint2x4*)&p[0] = *(uint2x4*)&state[0];
	*(uint2x4*)&p[2] = *(uint2x4*)&state[8];

#pragma unroll 4
	for (int i = 0; i < 4; i++){
		*(uint4*)&msg[i] = *(uint4*)&hash[i << 2];
		*(uint4*)&r[i << 2] = *(uint4*)&hash[i << 2];
	}
	r[16] = 0x80; r[17] = 0; r[18] = 0; r[19] = 0;
	r[20] = 0; r[21] = 0; r[22] = 0; r[23] = 0;
	r[24] = 0; r[25] = 0; r[26] = 0; r[27] = 0x02000000;
	r[28] = 0; r[29] = 0; r[30] = 0; r[31] = 0x02000000;
	y = p[1] ^ msg[0];
	__syncthreads();

	AES_ROUND_NOKEY(sharedMemory, &y);
	y ^= msg[1];
	AES_ROUND_NOKEY(sharedMemory, &y);
	y ^= msg[2];
	AES_ROUND_NOKEY(sharedMemory, &y);
	y ^= msg[3];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[0] ^= y;
	y = p[3];
	y.x ^= 0x80;
	AES_ROUND_NOKEY(sharedMemory, &y);
	AES_ROUND_NOKEY(sharedMemory, &y);
	y.w ^= 0x02000000;
	AES_ROUND_NOKEY(sharedMemory, &y);
	y.w ^= 0x02000000;
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;

	// 1
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	r[0] ^= 0x200;
	r[3] ^= 0xFFFFFFFF;
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;
	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	y = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);

	r[4] ^= r[29]; r[5] ^= r[30];
	r[6] ^= r[31]; r[7] ^= r[0];

	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	y = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);

	p[0] ^= y;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, y);


	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, y);

	// 2
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	r[7] ^= (~0x200);
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	y = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	y = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[0] ^= y;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, y);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, y);

	// 3
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	r[30] ^= 0x200;
	r[31] ^= 0xFFFFFFFF;
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;

	*(uint4*)&r[0] ^= *(uint4*)&r[25];
	y = p[3] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	r[4] ^= r[29];
	r[5] ^= r[30];
	r[6] ^= r[31];
	r[7] ^= r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[8] ^= *(uint4*)&r[1];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[12] ^= *(uint4*)&r[5];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[2] ^= y;
	*(uint4*)&r[16] ^= *(uint4*)&r[9];
	y = p[1] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[20] ^= *(uint4*)&r[13];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[24] ^= *(uint4*)&r[17];
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	*(uint4*)&r[28] ^= *(uint4*)&r[21];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[0] ^= y;

	/* round 3, 7, 11 */
	round_3_7_11(sharedMemory, r, p, y);

	/* round 4, 8, 12 */
	round_4_8_12(sharedMemory, r, p, y);

	/* round 13 */
	KEY_EXPAND_ELT(sharedMemory, &r[0]);
	*(uint4*)&r[0] ^= *(uint4*)&r[28];
	y = p[0] ^ *(uint4*)&r[0];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[4]);
	*(uint4*)&r[4] ^= *(uint4*)&r[0];
	y ^= *(uint4*)&r[4];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[8]);
	*(uint4*)&r[8] ^= *(uint4*)&r[4];
	y ^= *(uint4*)&r[8];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[12]);
	*(uint4*)&r[12] ^= *(uint4*)&r[8];
	y ^= *(uint4*)&r[12];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[3] ^= y;
	KEY_EXPAND_ELT(sharedMemory, &r[16]);
	*(uint4*)&r[16] ^= *(uint4*)&r[12];
	y = p[2] ^ *(uint4*)&r[16];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[20]);
	*(uint4*)&r[20] ^= *(uint4*)&r[16];
	y ^= *(uint4*)&r[20];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[24]);
	*(uint4*)&r[24] ^= *(uint4*)&r[20];
	r[25] ^= 0x200;
	r[27] ^= 0xFFFFFFFF;
	y ^= *(uint4*)&r[24];
	AES_ROUND_NOKEY(sharedMemory, &y);
	KEY_EXPAND_ELT(sharedMemory, &r[28]);
	*(uint4*)&r[28] ^= *(uint4*)&r[24];
	y ^= *(uint4*)&r[28];
	AES_ROUND_NOKEY(sharedMemory, &y);
	p[1] ^= y;

	*(uint2x4*)&hash[0] = *(uint2x4*)&state[0] ^ *(uint2x4*)&p[2];
	*(uint2x4*)&hash[8] = *(uint2x4*)&state[8] ^ *(uint2x4*)&p[0];
	//	}
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

	dim3 grid((threads + 128 - 1) / 128);
	dim3 block(128);

	x11_cubehashShavite512_gpu_hash_64 << <grid, block >> > (threads, d_hash);
}

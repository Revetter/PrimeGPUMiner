/*******************************************************************************************
   
 Nexus Earth 2018 

 (credits: cbuchner1 for sieving)

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php
  
*******************************************************************************************/

#include <ctype.h>
#include <stdio.h>

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cuda.h>
#include "cuda_runtime.h"

#include <map>
#include <sys/time.h>

#include "frame_resources.h"
#include "cuda.h"

#if _WIN32
#include <Winsock2.h> // for struct timeval
#endif

#include <cudaProfiler.h>

static __device__ uint64_t MAKE_ULONGLONG(uint32_t LO, uint32_t HI)
{
	#if __CUDA_ARCH__ >= 130
		return __double_as_longlong(__hiloint2double(HI, LO));
	#else
		return (uint64_t)LO | (((uint64_t)HI) << 32);
	#endif
}

template<uint32_t sharedSizeKB, uint32_t nThreadsPerBlock, uint32_t offsetsA>
__global__ void primesieve_kernelA(uint8_t *g_bit_array_sieve, uint32_t nBitArray_Size, uint4 *primes, uint16_t *prime_remainders, uint32_t *base_remainders, uint32_t nPrimorialEndPrime, uint32_t nPrimeLimitA);

__device__ __forceinline__
uint32_t mod_p_small(uint64_t a, uint32_t p, uint64_t recip) 
{
	uint64_t q = __umul64hi(a, recip);
	int64_t r = a - p*q;
	if (r >= p)
    r -= p;
	return (uint32_t)r;
}

int device_map[GPU_MAX] = {0,1,2,3,4,5,6,7};

extern "C" void cuda_reset_device()
{
	cudaDeviceReset();
}

extern "C" void cuda_device_synchronize()
{
  cudaDeviceSynchronize();
}

extern "C" int cuda_num_devices()
{
    int version;
    cudaError_t err = cudaDriverGetVersion(&version);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Unable to query CUDA driver version! Is an nVidia driver installed?");
        exit(1);
    }

    int maj = version / 1000, min = version % 100; // same as in deviceQuery sample
    if (maj < 5 || (maj == 5 && min < 5))
    {
        fprintf(stderr, "Driver does not support CUDA %d.%d API! Update your nVidia driver!", 5, 5);
        exit(1);
    }

    int GPU_N;
    err = cudaGetDeviceCount(&GPU_N);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Unable to query number of CUDA devices! Is an nVidia driver installed?");
        exit(1);
    }
    return GPU_N;
}


extern "C" const char* cuda_devicename(int index)
{
	const char *result = NULL;
	cudaDeviceProp props;
	if (cudaGetDeviceProperties(&props, index) == cudaSuccess)
		result = strdup(props.name);

	return result;
}



extern "C" void cuda_init(int thr_id)
{
  fprintf(stderr, "thread %d maps to CUDA device #%d\n", thr_id, device_map[thr_id]);

  CUcontext ctx;
  cuCtxCreate( &ctx, CU_CTX_SCHED_AUTO, device_map[thr_id] );
  cuCtxSetCurrent(ctx);

  cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
}

extern "C" void cuda_free(int deviceCount)
{
  cuProfilerStop();

  for(int i = 0; i < deviceCount; ++i)
  {
    printf("\nDevice %d shutting down...", device_map[i]);

    cudaSetDevice(device_map[i]);
    cudaDeviceSynchronize();
    
    cudaDeviceReset();
  }
  printf("\n");
}


__constant__ uint64_t c_zTempVar[17];
__constant__ uint32_t c_offsetsA[32];
__constant__ uint32_t c_offsetsB[32];


__constant__ uint16_t c_primes[512];
__constant__ uint16_t c_blockoffset_mod_p[62][512];
uint4 *d_primesInverseInvk[GPU_MAX];
uint32_t *d_base_remainders[GPU_MAX];

cudaStream_t d_sieveA_Stream[GPU_MAX];
cudaStream_t d_sieveB_Stream[GPU_MAX];
cudaStream_t d_compact_Stream[GPU_MAX];

cudaEvent_t d_sieveA_Event[GPU_MAX][FRAME_COUNT];
cudaEvent_t d_sieveB_Event[GPU_MAX][FRAME_COUNT];
cudaEvent_t d_compact_Event[GPU_MAX][FRAME_COUNT];

struct FrameResource frameResources[GPU_MAX];


static uint32_t nOffsetsA;
static uint32_t nOffsetsB;


extern "C" void cuda_set_zTempVar(uint32_t thr_id, const uint64_t *limbs)
{
    CHECK(cudaMemcpyToSymbol(c_zTempVar, limbs, 17*sizeof(uint64_t), 0, cudaMemcpyHostToDevice));
}

extern "C" void cuda_set_primes(uint32_t thr_id, uint32_t *primes, 
                                uint32_t *inverses, uint64_t *invk, 
                                uint32_t nPrimeLimit, 
                                uint32_t nBitArray_Size, 
                                uint32_t sharedSizeKB,
                                uint32_t nPrimorialEndPrime,
                                uint32_t nPrimeLimitA)
{ 

  uint32_t primeinverseinvk_size = sizeof(uint32_t) * 4 * nPrimeLimit;
  
  uint32_t nonce64_size = OFFSETS_MAX * sizeof(uint64_t);
  uint32_t nonce32_size = OFFSETS_MAX * sizeof(uint32_t);

  uint32_t sharedSizeBits = sharedSizeKB * 1024 * 8;
  uint32_t allocSize = ((nBitArray_Size + sharedSizeBits - 1) / sharedSizeBits) * sharedSizeBits;
  uint32_t bitarray_size = (allocSize+31)/32 * sizeof(uint32_t);
  uint32_t remainder_size = 512 * sizeof(uint16_t) * 16;

  //uint32_t d_frameBytes = (BUFFER_COUNT + 1)*(nonce64_size + nonce32_size + sizeof(uint32_t)) +
                           //bitarray_size + remainder_size;
  //printf("Device KB per Frame: %d\n", d_frameBytes >> 10);

	uint32_t *lst = (uint32_t*)malloc(primeinverseinvk_size);
	CHECK(cudaMalloc(&d_primesInverseInvk[thr_id],  primeinverseinvk_size));
	for (int i = 0; i < nPrimeLimit; ++i)
	{
		memcpy(&lst[i * 4 + 0], &primes[i], sizeof(uint32_t));
		memcpy(&lst[i * 4 + 1], &inverses[i], sizeof(uint32_t));
		memcpy(&lst[i * 4 + 2], &invk[i], sizeof(uint64_t));
	}
	CHECK(cudaMemcpy(d_primesInverseInvk[thr_id], lst, primeinverseinvk_size, cudaMemcpyHostToDevice));
	free(lst);
	
  CHECK(cudaMalloc(&d_base_remainders[thr_id],  nPrimeLimit * sizeof(uint32_t)));




    //these will need thier own frame index so they can be parallelized
  //printf("bitarray_size: %d KB\n", bitarray_size >> 10);

  for(int i = 0; i < FRAME_COUNT; ++i)
  {
      //test
    CHECK(cudaMalloc(&frameResources[thr_id].d_result_offsets[i], nonce64_size));
    CHECK(cudaMallocHost(&frameResources[thr_id].h_result_offsets[i], nonce64_size));
    CHECK(cudaMalloc(&frameResources[thr_id].d_result_meta[i], nonce32_size));
    CHECK(cudaMallocHost(&frameResources[thr_id].h_result_meta[i], nonce32_size));
    CHECK(cudaMalloc(&frameResources[thr_id].d_result_count[i], sizeof(uint32_t)));
    CHECK(cudaMallocHost(&frameResources[thr_id].h_result_count[i], sizeof(uint32_t)));

      //test stats
    CHECK(cudaMalloc(&frameResources[thr_id].d_primes_checked[i], sizeof(uint32_t)));
    CHECK(cudaMallocHost(&frameResources[thr_id].h_primes_checked[i], sizeof(uint32_t)));
    CHECK(cudaMalloc(&frameResources[thr_id].d_primes_found[i], sizeof(uint32_t)));
    CHECK(cudaMallocHost(&frameResources[thr_id].h_primes_found[i], sizeof(uint32_t)));

      //compaction
    CHECK(cudaMalloc(&frameResources[thr_id].d_nonce_offsets[i], nonce64_size * BUFFER_COUNT));
    CHECK(cudaMalloc(&frameResources[thr_id].d_nonce_meta[i],    nonce32_size * BUFFER_COUNT));  
    CHECK(cudaMalloc(&frameResources[thr_id].d_nonce_count[i],  sizeof(uint32_t) * BUFFER_COUNT));
    CHECK(cudaMallocHost(&frameResources[thr_id].h_nonce_count[i], sizeof(uint32_t)));

      //sieving
    CHECK(cudaMalloc(&frameResources[thr_id].d_prime_remainders[i], remainder_size));
    CHECK(cudaMalloc(&frameResources[thr_id].d_bit_array_sieve[i], bitarray_size));
  }
  
  uint32_t nBlocks = (nBitArray_Size + sharedSizeBits-1) / sharedSizeBits;
  
  uint16_t p[512];
  for(uint32_t i = 0; i < nPrimeLimitA; ++i)
    p[i] = primes[i];

  CHECK(cudaMemcpyToSymbol(c_primes, 
                           p, 
                           nPrimeLimitA * sizeof(uint16_t), 
                           0, 
                           cudaMemcpyHostToDevice));

  //printf("nBlocks = %d\n", nBlocks);
  for (int block = 0; block < nBlocks; ++block)
  {
	  uint32_t blockOffset = sharedSizeBits * block;
	  uint16_t offsets[512];

	  for (int i = 0; i < nPrimeLimitA; ++i)
		  offsets[i] = primes[i] - (blockOffset % primes[i]);

	  CHECK(cudaMemcpyToSymbol(c_blockoffset_mod_p, 
                             offsets, 
                             nPrimeLimitA * sizeof(uint16_t), 
                             block*512*sizeof(uint16_t), 
                             cudaMemcpyHostToDevice));
  }


  //printf("Thread %d initialized\n", thr_id);
  CHECK(cudaGetLastError());
}

__device__ uint32_t mpi_mod_int(uint64_t *A, uint32_t B, uint64_t recip)
{
	if (B == 1)
		return 0;
	else if (B == 2)
		return A[0]&1;

	int i;
	uint64_t x,y;

#pragma unroll 16
	for( i = 16, y = 0; i > 0; --i)
	{
		x  = A[i-1];
		y = (y << 32) | (x >> 32);
		y = mod_p_small(y, B, recip);
		
		x <<= 32;		
		y = (y << 32) | (x >> 32);
		y = mod_p_small(y, B, recip);
	}

	return (uint32_t)y;
}

__global__ void base_remainders_kernel(uint4 *g_primes, uint32_t *g_base_remainders, 
                                       uint32_t nPrimorialEndPrime, uint32_t nPrimeLimit)
{
	uint32_t i = nPrimorialEndPrime + blockDim.x * blockIdx.x + threadIdx.x;
	if (i < nPrimeLimit)
	{
		uint4 tmp = g_primes[i];
		uint64_t rec = MAKE_ULONGLONG(tmp.z, tmp.w);
		g_base_remainders[i] = mpi_mod_int(c_zTempVar, tmp.x, rec);
	}
}

extern "C" void cuda_compute_base_remainders(uint32_t thr_id, 
                                             uint32_t nPrimorialEndPrime, 
                                             uint32_t nPrimeLimit)
{
    int nThreads = nPrimeLimit - nPrimorialEndPrime;
    int nThreadsPerBlock = 256;
    int nBlocks = (nThreads + nThreadsPerBlock-1) / nThreadsPerBlock;

    dim3 block(nThreadsPerBlock);
    dim3 grid(nBlocks);

    base_remainders_kernel<<<grid, block, 0>>>(d_primesInverseInvk[thr_id], 
                                               d_base_remainders[thr_id], 
                                               nPrimorialEndPrime, 
                                               nPrimeLimit);

}

template<int Begin, int End, int Step = 1>
struct Unroller {
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action) {
        action(Begin);
        Unroller<Begin+Step, End, Step>::step(action);
    }
};

template<int End, int Step>
struct Unroller<End, End, Step> {
    template<typename Action>
    __device__ __forceinline__ static void step(Action& action) {
    }
};


__global__ void primesieve_kernelA0(uint32_t thr_id, uint64_t origin, 
uint32_t sharedSizeKB, uint4 *primes, uint32_t *base_remainders, 
uint16_t *prime_remainders, uint32_t nPrimorialEndPrime, 
uint32_t nPrimeLimitA, uint32_t offsetsA)
{	
	uint32_t position = (blockIdx.x << 9) + threadIdx.x;
	uint32_t memory_position = position << 4;
	uint4 tmp = primes[threadIdx.x];
	uint64_t rec = MAKE_ULONGLONG(tmp.z, tmp.w);
	uint32_t a, c;

    // note that this kernel currently hardcodes 9 specific offsets
  a = mod_p_small(origin + base_remainders[threadIdx.x], tmp.x, rec);
  c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	  prime_remainders[memory_position] = c;

  //#pragma unroll
  for(uint32_t i = 1; i < offsetsA; ++i)
  {
    a += (c_offsetsA[i] - c_offsetsA[i-1]);
    if(a >= tmp.x) 
      a -= tmp.x;	
	  c = mod_p_small((uint64_t)(tmp.x - a)*tmp.y, tmp.x, rec);
	  prime_remainders[memory_position + i] = c;
  }
}

template<uint32_t sharedSizeKB, uint32_t nThreadsPerBlock, uint32_t offsetsA>
__global__ void primesieve_kernelA(uint8_t *g_bit_array_sieve, uint32_t nBitArray_Size, uint4 *primes, uint16_t *prime_remainders, uint32_t *base_remainders, uint32_t nPrimorialEndPrime, uint32_t nPrimeLimitA)
{
	extern __shared__ uint32_t shared_array_sieve[];

  uint32_t sharedSizeBytes = sharedSizeKB<<10;
	uint32_t sizeSieve = sharedSizeBytes << 3;
  uint32_t i, j;

	//if (sharedSizeKB == 32 && nThreadsPerBlock == 1024)
  //{
//#pragma unroll 8
	//	for (int i=0; i <  8; ++i) 
  //    shared_array_sieve[threadIdx.x+i*1024] = 0;
	//}
	//else if (sharedSizeKB == 48 && nThreadsPerBlock == 768) 
  //{
//#pragma unroll 16
	//	for (i=0; i < 16; ++i) 
  //    shared_array_sieve[threadIdx.x+i*768] = 0;
	//}
	//else if (sharedSizeKB == 32 && nThreadsPerBlock == 512) 
  //{
#pragma unroll 16
		for (int i=0; i < 16; ++i) 
      shared_array_sieve[threadIdx.x+i*512] = 0;
	//}
	__syncthreads();
		
	for (i = nPrimorialEndPrime; i < nPrimeLimitA; ++i)
	{
		uint32_t pr = c_primes[i];
		uint32_t pre2 = c_blockoffset_mod_p[blockIdx.x][i];

		// precompute pIdx, nAdd
		uint32_t pIdx = threadIdx.x * pr;
		uint32_t nAdd;
		//if (nThreadsPerBlock == 1024)     
    //  nAdd = pr << 10;
		//else if (nThreadsPerBlock == 768) 
    //  nAdd = (pr << 9) + (pr << 8);
		//else if (nThreadsPerBlock == 512) 
      nAdd = pr << 9;
		//else    
    //  nAdd = pr * nThreadsPerBlock;

		uint32_t pre1[offsetsA];
		auto pre = [&pre1, &prime_remainders, &i](uint32_t o)
    {
			pre1[o] = prime_remainders[(i << 4) + o]; // << 4 because we have space for 16 offsets
		};

		Unroller<0, offsetsA>::step(pre);

		auto loop = [&sizeSieve, &pIdx, &nAdd, &prime_remainders, &pre1, &pre2, &pr](uint32_t o)
    {
			uint32_t tmp = pre1[o] + pre2;
			tmp = (tmp >= pr) ? (tmp - pr) : tmp;
			for(uint32_t index = tmp + pIdx; index < sizeSieve; index+= nAdd)				
				atomicOr(&shared_array_sieve[index >> 5], 1 << (index & 31));
		};

		Unroller<0, offsetsA>::step(loop);
	}

	__syncthreads();
	g_bit_array_sieve += sharedSizeBytes * blockIdx.x;

//	if (sharedSizeKB == 32 && nThreadsPerBlock == 1024) 
//  {
//#pragma unroll 8
//		for (int i = 0; i < 8192; i+=1024) // fixed value
//			((uint32_t*)g_bit_array_sieve)[threadIdx.x+i] |= shared_array_sieve[threadIdx.x+i];
//	}
//	else if (sharedSizeKB == 48 && nThreadsPerBlock == 768) 
//  {
//#pragma unroll 16
//		for (i = 0; i < 12288; i+=768)
//    { // fixed value
//      j = threadIdx.x+i;
//			((uint32_t*)g_bit_array_sieve)[j] = shared_array_sieve[j];
//    }
//	}
//	else if (sharedSizeKB == 32 && nThreadsPerBlock == 512) 
 // {
#pragma unroll 16
		for (int i = 0; i < 8192; i += 512) // fixed value
		{	
      j = threadIdx.x + i;
      ((uint32_t*)g_bit_array_sieve)[j] = shared_array_sieve[j];
    }
//}
}

__global__ void primesieve_kernelB(uint8_t *g_bit_array_sieve, 
                                   uint32_t nBitArray_Size, 
                                   uint4 *primes, 
                                   uint32_t *base_remainders, 
                                   uint64_t base_offset, 
                                   uint32_t nPrimorialEndPrime, 
                                   uint32_t nPrimeLimit,
                                   uint32_t offsetsB)
{
	uint32_t position = blockDim.x * blockIdx.x + threadIdx.x;
	uint32_t i = nPrimorialEndPrime + (position >> 3);
	uint32_t o = position & 0x7; //can only be one of 8 offsets
	if ( i < nPrimeLimit && o < offsetsB)
	{
		uint32_t *g_word_array_sieve = (uint32_t*)g_bit_array_sieve;
		uint4 tmp = primes[i];

		uint32_t p = tmp.x;
		uint32_t inv = tmp.y;
		uint64_t recip = MAKE_ULONGLONG(tmp.z, tmp.w);
		uint32_t remainder = mod_p_small(base_offset + base_remainders[i] + c_offsetsB[o], p, recip);
		uint32_t index = mod_p_small((uint64_t)(p - remainder)*inv, p, recip);

		for (; index < nBitArray_Size; index += p)
		{
			atomicOr(&g_word_array_sieve[index >> 5], 1 << (index & 31));
		}
	}
}

void kernelA_launcher(uint32_t thr_id,
                      uint32_t frame_index, 
                      uint32_t sharedSizeKB, 
                      uint32_t nThreadsPerBlock, 
                      uint32_t nBitArray_Size,   
                      uint32_t nPrimorialEndPrime,
                      uint32_t nPrimeLimitA)
{
  const int sharedSizeBits = sharedSizeKB * 1024 * 8;
	int nBlocks = (nBitArray_Size + sharedSizeBits-1) / sharedSizeBits;


	dim3 block(nThreadsPerBlock);
	dim3 grid(nBlocks);

  //printf("primesieve_kernelA<%d, %d, %d><<<%d, %d, %d>>>\n", sharedSizeKB, nThreadsPerBlock, nOffsetsA, grid.x, block.x, sharedSizeBits/8);

	if (sharedSizeKB == 32 && nThreadsPerBlock == 1024)
	{
    switch(nOffsetsA)
    {
      case 1:
		primesieve_kernelA<32, 1024, 1><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 2:
		primesieve_kernelA<32, 1024, 2><<<grid, block, sharedSizeBits/8>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 3:
		primesieve_kernelA<32, 1024, 3><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 4:
		primesieve_kernelA<32, 1024, 4><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 5:
		primesieve_kernelA<32, 1024, 5><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 6:
		primesieve_kernelA<32, 1024, 6><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 7:
		primesieve_kernelA<32, 1024, 7><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 8:
		primesieve_kernelA<32, 1024, 8><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 9:
		primesieve_kernelA<32, 1024, 9><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 10:
		primesieve_kernelA<32, 1024, 10><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 11:
		primesieve_kernelA<32, 1024, 11><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 12:
		primesieve_kernelA<32, 1024, 12><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
    }
	}
	else if (sharedSizeKB == 48 && nThreadsPerBlock == 768)
	{
    switch(nOffsetsA)
    {
      case 1:
		primesieve_kernelA<48, 768, 1><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 2:
		primesieve_kernelA<48, 768, 2><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 3:
		primesieve_kernelA<48, 768, 3><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 4:
		primesieve_kernelA<48, 768, 4><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 5:
		primesieve_kernelA<48, 768, 5><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 6:
		primesieve_kernelA<48, 768, 6><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 7:
		primesieve_kernelA<48, 768, 7><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 8:
		primesieve_kernelA<48, 768, 8><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 9:
		primesieve_kernelA<48, 768, 9><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 10:
		primesieve_kernelA<48, 768, 10><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 11:
		primesieve_kernelA<48, 768, 11><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 12:
		primesieve_kernelA<48, 768, 12><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
    }
	}
	else if (sharedSizeKB == 32 && nThreadsPerBlock == 512)
	{
		    switch(nOffsetsA)
    {
      case 1:
		primesieve_kernelA<32, 512, 1><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 2:
		primesieve_kernelA<32, 512, 2><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 3:
		primesieve_kernelA<32, 512, 3><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 4:
		primesieve_kernelA<32, 512, 4><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 5:
		primesieve_kernelA<32, 512, 5><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 6:
		primesieve_kernelA<32, 512, 6><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 7:
		primesieve_kernelA<32, 512, 7><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 8:
		primesieve_kernelA<32, 512, 8><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 9:
		primesieve_kernelA<32, 512, 9><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 10:
		primesieve_kernelA<32, 512, 10><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 11:
		primesieve_kernelA<32, 512, 11><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
      case 12:
		primesieve_kernelA<32, 512, 12><<<grid, block, sharedSizeBits/8, d_sieveA_Stream[thr_id]>>>(frameResources[thr_id].d_bit_array_sieve[frame_index], nBitArray_Size, d_primesInverseInvk[thr_id], frameResources[thr_id].d_prime_remainders[frame_index], d_base_remainders[thr_id], nPrimorialEndPrime, nPrimeLimitA);
      break;
    }
	}
	else
	{
		fprintf(stderr, "Unsupported Shared Mem / Block size configuration for KernelA! Use 32kb, 1024 threads or 48kb, 768 threads!\n");
		exit(1);
	}
}

extern "C" void cuda_set_offsets(uint32_t thr_id, uint32_t *OffsetsA, uint32_t A_count,
                                                  uint32_t *OffsetsB, uint32_t B_count)
{
    nOffsetsA = A_count;
    nOffsetsB = B_count;

    if(nOffsetsA > 12)
      exit(1);
    if(nOffsetsB > 8)
      exit(1);

    CHECK(cudaMemcpyToSymbol(c_offsetsA, OffsetsA, 
                      nOffsetsA*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

    CHECK(cudaMemcpyToSymbol(c_offsetsB, OffsetsB, 
                      nOffsetsB*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

__global__ void compact_offsets(uint64_t *d_nonce_offsets, uint32_t *d_nonce_meta, uint32_t *d_nonce_count, 
                                uint8_t *d_bit_array_sieve, uint32_t max_count, uint32_t sieve_index)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx < max_count)
  {
    uint32_t *d_word_array_sieve = (uint32_t *)d_bit_array_sieve;
    uint64_t nonce_offset = static_cast<uint64_t>(max_count) * 
                            static_cast<uint64_t>(sieve_index) + idx;

    if((d_word_array_sieve[idx >> 5] & (1 << (idx & 31))) == 0)
    {
      atomicMin(d_nonce_count, OFFSETS_MAX);
      uint32_t i = atomicAdd(d_nonce_count, 1);
      
      if(i < OFFSETS_MAX)
      {
        d_nonce_offsets[i] = nonce_offset;
        d_nonce_meta[i] = 0;
      }
    }
  }
}


extern "C" bool cuda_compute_primesieve(uint32_t thr_id,
                                        uint32_t thr_count, 
                                        uint32_t nSharedSizeKB,
                                        uint32_t nThreadsKernelA, 
                                        uint64_t base_offset,
                                        uint64_t primorial, 
                                        uint32_t nPrimorialEndPrime, 
                                        uint32_t nPrimeLimitA, 
                                        uint32_t nPrimeLimitB, 
                                        uint32_t nBitArray_Size, 
                                        uint32_t nDifficulty,
                                        uint32_t sieve_index,
                                        uint32_t test_index)
{
  int nThreads = nPrimeLimitB - nPrimeLimitA;
  int nThreadsPerBlock = 32 * 8;        
  int nBlocks = (nThreads + nThreadsPerBlock - 1) / (nThreadsPerBlock / 8);

  dim3 block(nThreadsPerBlock);
  dim3 grid(nBlocks);
  dim3 block2(nPrimeLimitA);
	dim3 grid2(1);
  dim3 block3(256);
  dim3 grid3((nBitArray_Size + block.x - 1) / block.x);

  uint32_t curr_sieve = sieve_index % FRAME_COUNT;
  uint32_t curr_test = test_index % FRAME_COUNT;

  if (nBlocks == 0)
  {
    printf("error. no blocks.\n");
    return false;
  }

  if(cudaEventQuery(d_compact_Event[thr_id][curr_sieve]) == cudaErrorNotReady)
    return false;

  sieve_index = sieve_index * thr_count + thr_id;
  uint64_t base_offsetted = base_offset + primorial * (uint64_t)nBitArray_Size * (uint64_t)sieve_index;

  if(base_offsetted >= 0xF000000000000000)
    printf("Search Range Almost Exhausted!!! Try using a smaller Primorial\n");
  
    //precompute prime remainders for this origin base offset
	primesieve_kernelA0<<<grid2, block2, 0, d_sieveA_Stream[thr_id]>>>(
     thr_id, base_offsetted, nSharedSizeKB, 
     d_primesInverseInvk[thr_id], d_base_remainders[thr_id], 
     frameResources[thr_id].d_prime_remainders[curr_sieve], 
     nPrimorialEndPrime, nPrimeLimitA, nOffsetsA);
  
    //sieve with shared memory first N primes
 	kernelA_launcher(thr_id, curr_sieve, nSharedSizeKB, nThreadsKernelA, 
                   nBitArray_Size, nPrimorialEndPrime, nPrimeLimitA);

  CHECK(cudaEventRecord(d_sieveA_Event[thr_id][curr_sieve], d_sieveA_Stream[thr_id]));
  
    //sieveB must wait on sieveA
  CHECK(cudaStreamWaitEvent(d_sieveB_Stream[thr_id], d_sieveA_Event[thr_id][curr_sieve], 0));

    //sieve with remainder of primes
  primesieve_kernelB<<<grid, block, 0, d_sieveB_Stream[thr_id]>>>(
      frameResources[thr_id].d_bit_array_sieve[curr_sieve],
      nBitArray_Size,  
      d_primesInverseInvk[thr_id],
      d_base_remainders[thr_id],
      base_offsetted, 
      nPrimeLimitA, 
      nPrimeLimitB,
      nOffsetsB);

  CHECK(cudaEventRecord(d_sieveB_Event[thr_id][curr_sieve], d_sieveB_Stream[thr_id]));


    //make sure sieving is finished before compacting
  CHECK(cudaStreamWaitEvent(d_compact_Stream[thr_id], d_sieveB_Event[thr_id][curr_sieve],  0));

  compact_offsets<<<grid3, block3, 0, d_compact_Stream[thr_id]>>>(
      frameResources[thr_id].d_nonce_offsets[curr_test],
      frameResources[thr_id].d_nonce_meta[curr_test], 
      frameResources[thr_id].d_nonce_count[curr_test], 
      frameResources[thr_id].d_bit_array_sieve[curr_sieve],
      nBitArray_Size,
      sieve_index);
  
  CHECK(cudaMemcpyAsync(frameResources[thr_id].h_nonce_count[curr_test], 
                        frameResources[thr_id].d_nonce_count[curr_test], 
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, d_compact_Stream[thr_id]));

  CHECK(cudaEventRecord(d_compact_Event[thr_id][curr_sieve], d_compact_Stream[thr_id]));

  return true;
}

extern "C" void cuda_wait_sieve(uint32_t thr_id, uint32_t sieve_index)
{
  uint32_t curr_sieve = sieve_index % FRAME_COUNT;
  CHECK(cudaEventSynchronize(d_compact_Event[thr_id][curr_sieve]));
}

extern "C" void cuda_init_sieve(uint32_t thr_id)
{
  CHECK(cudaStreamCreateWithFlags(&d_sieveA_Stream[thr_id], cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&d_sieveB_Stream[thr_id], cudaStreamNonBlocking));
  CHECK(cudaStreamCreateWithFlags(&d_compact_Stream[thr_id], cudaStreamNonBlocking));

  for(int i = 0; i < FRAME_COUNT; ++i)
  {
    CHECK(cudaEventCreateWithFlags(&d_sieveA_Event[thr_id][i],   cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&d_sieveB_Event[thr_id][i],   cudaEventDisableTiming));
    CHECK(cudaEventCreateWithFlags(&d_compact_Event[thr_id][i],   cudaEventDisableTiming | cudaEventBlockingSync));
  }
}

extern "C" void cuda_free_sieve(uint32_t thr_id)
{
  CHECK(cudaStreamDestroy(d_sieveA_Stream[thr_id]));
  CHECK(cudaStreamDestroy(d_sieveB_Stream[thr_id]));
  CHECK(cudaStreamDestroy(d_compact_Stream[thr_id]));
  
  for(int i = 0; i < FRAME_COUNT; ++i)
  {
    CHECK(cudaEventDestroy(d_sieveA_Event[thr_id][i]));
    CHECK(cudaEventDestroy(d_sieveB_Event[thr_id][i]));
    CHECK(cudaEventDestroy(d_compact_Event[thr_id][i]));
  }
}

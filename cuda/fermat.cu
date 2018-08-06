/*******************************************************************************************
   
 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php
  
*******************************************************************************************/
#include "fermat.h"
#include "frame_resources.h"

extern struct FrameResource frameResources[GPU_MAX];

__constant__ uint32_t c_zFirstSieveElement[WORD_MAX];
__constant__ uint32_t c_quit;


cudaStream_t d_fermat_Stream[GPU_MAX];
cudaEvent_t d_fermat_Event[GPU_MAX][FRAME_COUNT];

extern cudaEvent_t d_compact_Event[GPU_MAX][FRAME_COUNT];
__constant__ uint32_t c_offsets[32];
static uint32_t nOffsetsTest;

__host__ __device__ void print_hex(uint32_t *p)
{
  for(int i = 0; i < WORD_MAX; ++i)
    printf("%08X\n", p[i]);

  printf("\n");
}

__device__ void print_hex2(uint32_t *a, uint32_t *b)
{
  for(int j = 0; j < WORD_MAX; ++j)
  {
   char c = a[j] > b[j] ? '>' : a[j] == b[j] ? '=' : '<';

    printf("%d\t%08X", j, a[j]);
    printf("\t%c", c);
    printf("\t%08X\n", b[j]);
  }
  printf("\n");
}

extern "C" void cuda_set_FirstSieveElement(uint32_t thr_id, uint32_t *limbs)
{
  CHECK(cudaMemcpyToSymbol(c_zFirstSieveElement, limbs, 
    WORD_MAX*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}

extern "C" void cuda_set_quit(uint32_t quit)
{
  CHECK(cudaMemcpyToSymbol(c_quit, &quit,
                           sizeof(uint32_t), 0, cudaMemcpyHostToDevice));
}


struct uint1024
{
  uint32_t array[WORD_MAX];
};

__device__ void assign(uint32_t *l, uint32_t *r)
{
  #pragma unroll
  for(int i = 0; i < WORD_MAX; ++i)
    l[i] = r[i];
}

__device__ int inv2adic(uint32_t x)
{
  uint32_t a;
  a = x;
  x = (((x+2)&4)<<1)+x;
  x *= 2 - a*x;
  x *= 2 - a*x;
  x *= 2 - a*x;
  return -x;
}

__device__ uint32_t cmp_ge_n(uint32_t *x, uint32_t *y)
{
  for(int i = WORD_MAX-1; i >= 0; --i)
  {
    if(x[i] > y[i])
      return 1;

    if(x[i] < y[i])
      return 0;
  }
  return 1;
}

__device__ uint32_t sub_n(uint32_t *z, uint32_t *x, uint32_t *y)
{
  uint32_t temp;
  uint32_t c = 0;

  #pragma unroll
  for(int i = 0; i < WORD_MAX; ++i)
  {
    temp = x[i] - y[i] - c;
    c = (temp > x[i]);
    z[i] = temp;
  }
  return c;
}

__device__ void sub_ui(uint32_t *z, uint32_t *x, const uint32_t &ui)
{
  uint32_t temp = x[0] - ui;
  uint32_t c = temp > x[0];
  z[0] = temp;

  #pragma unroll
  for(int i = 1; i < WORD_MAX; ++i)
  {
    temp = x[i] - c;
    c = (temp > x[i]);
    z[i] = temp;
  }
}

__device__ void add_ui(uint32_t *z, uint32_t *x, const uint64_t &ui)
{
  uint32_t temp = x[0] + static_cast<uint32_t>(ui & 0xFFFFFFFF);
  uint32_t c = temp < x[0];
  z[0] = temp;

  temp = x[1] + static_cast<uint32_t>(ui >> 32) + c;
  c = temp < x[1];
  z[1] = temp;

  #pragma unroll
  for(int i = 2; i < WORD_MAX; ++i)
  {
    temp = x[i] + c;
    c = (temp < x[i]);
    z[i] = temp;
  }
}

__device__ uint32_t addmul_1(uint32_t *z, uint32_t *x, uint32_t y)
{ 
  uint64_t prod;
  uint32_t c = 0;

  #pragma unroll
  for(int i = 0; i < WORD_MAX; ++i)
  {
    prod = static_cast<uint64_t>(x[i]) * static_cast<uint64_t>(y);
    prod += c;
    prod += z[i];
    z[i] = prod;
    c = prod >> 32;    
  }
  
  return c;
}

__device__ void mulredc(uint32_t *z, uint32_t *x, uint32_t *y, uint32_t *n, const uint32_t d, uint32_t *t)
{
  int i, j;
  uint32_t m, c;
  uint64_t temp;
  
  #pragma unroll
  for(i = 0; i < WORD_MAX + 2; ++i)
    t[i] = 0;
    
  for(i = 0; i < WORD_MAX; ++i)
  {
    c = addmul_1(t, x, y[i]);
    
    temp = static_cast<uint64_t>(t[WORD_MAX]) + c;
    t[WORD_MAX] = temp;
    //t[WORD_MAX + 1] = temp >> 32;
    
    m = t[0]*d;
    
    c = addmul_1(t, n, m);
    temp = static_cast<uint64_t>(t[WORD_MAX]) + c;
    t[WORD_MAX] = temp;
    //t[WORD_MAX + 1] = temp >> 32;
    
    #pragma unroll
    for(j = 0; j <= WORD_MAX; ++j)
      t[j] = t[j+1];
  }
  if(cmp_ge_n(t, n))
    sub_n(t, t, n);
      
  #pragma unroll
  for(i = 0; i < WORD_MAX; ++i)
    z[i] = t[i];
}

__device__ void redc(uint32_t *z, uint32_t *x, uint32_t *n, const uint32_t d, uint32_t *t)
{
  int i, j;
  uint32_t m;
  
  #pragma unroll
  for(i = 0; i < WORD_MAX; ++i)
    t[i] = x[i];
    
  t[WORD_MAX] = 0;
  
  for(i = 0; i < WORD_MAX; ++i)
  {
    m = t[0]*d;
    t[WORD_MAX] = addmul_1(t, n, m);
    
    for(j = 0; j < WORD_MAX; ++j)
      t[j] = t[j+1];
      
    t[WORD_MAX] = 0;
  }
  
  if(cmp_ge_n(t, n))
    sub_n(t, t, n);
  
  #pragma unroll
  for(i = 0; i < WORD_MAX; ++i)
    z[i] = t[i];
}

__device__ uint32_t bit_count(uint32_t *x)
{
   uint32_t msb = 0; //most significant bit

   int bits = WORD_MAX << 5;
   int i;
   
   #pragma unroll
   for(i = 0; i < bits; ++i)
   {
     if(x[i>>5] & (1 << (i & 31)))
       msb = i;
   }
   
   return msb + 1; //any number will have at least 1-bit
}

__device__ void lshift(uint32_t *r, uint32_t *a, uint32_t shift)
{
  int i;

  #pragma unroll
  for(i = 0; i < WORD_MAX; ++i)
    r[i] = 0;
    
  uint32_t k = shift >> 5;
  shift = shift & 31;
  
  for(i = 0; i < WORD_MAX; ++i)
  {
    uint32_t ik = i + k;
    uint32_t ik1 = ik + 1;
    
    if(ik1 < WORD_MAX && shift != 0)
      r[ik1] |= (a[i] >> (32-shift));
    if(ik < WORD_MAX)
      r[ik] |= (a[i] << shift);
  }
}

__device__ void rshift(uint32_t *r, uint32_t *a, uint32_t shift)
{
  int i;
  
  #pragma unroll
  for(i = 0; i < WORD_MAX; ++i)
    r[i] = 0;
    
  uint32_t k = shift >> 5;
  shift = shift & 31;
  
  for(i = 0; i < WORD_MAX; ++i)
  {
    int ik = i - k;
    int ik1 = ik - 1;
    
    if(ik1 >= 0 && shift != 0)
      r[ik1] |= (a[i] << (32-shift));
    if(ik >= 0)
      r[ik] |= (a[i] >> shift);
  }
}

__device__ void lshift1(uint32_t *r, uint32_t *a)
{
  uint32_t i;
  uint32_t t = a[0];
  uint32_t t2;
  
  r[0] = t << 1;
  for(i = 1; i < WORD_MAX; ++i)
  {
    t2 = a[i];
    r[i] = (t2 << 1) | (t >> 31);
    t = t2;
  } 
}

__device__ void rshift1(uint32_t *r, uint32_t *a)
{
  int i;
  uint32_t n = WORD_MAX-1;
  uint32_t t = a[n];
  uint32_t t2;
  
  r[n] = t >> 1;
  for(i = n-1; i >= 0; --i)
  {
    t2 = a[i];
    r[i] = (t2 >> 1) | (t << 31);
    t = t2;
  }
}

__device__ void calcBar(uint32_t *a, uint32_t *b, uint32_t *n, uint32_t *t)
{
    #pragma unroll
    for(int i = 0; i < WORD_MAX; ++i)
        a[i] = 0;
      
    lshift(t, n, (WORD_MAX<<5) - bit_count(n));
    sub_n(a, a, t);
    
      //calculate R mod N;
    while(cmp_ge_n(a, n))
    {
        rshift1(t, t);
        if(cmp_ge_n(a, t))
          sub_n(a, a, t);
    }
    
      //calculate 2R mod N;
    lshift1(b, a);
    if(cmp_ge_n(b, n))
      sub_n(b, b, n);
}

__device__ void pow2m(uint32_t *X, uint32_t *Exp, uint32_t *N)
{
  uint32_t A[WORD_MAX];
  uint32_t d;
  uint32_t t[WORD_MAX + 1];
  
  d = inv2adic(N[0]);

  calcBar(X, A, N, t);

  for(int i = bit_count(Exp)-1; i >= 0; --i)
  {
    mulredc(X, X, X, N, d, t); 

    if(Exp[i>>5] & (1 << (i & 31)))
      mulredc(X, X, A, N, d, t);

  }
  redc(X, X, N, d, t);
}


__device__ bool fermat_prime(uint32_t *p)
{
  uint32_t e[WORD_MAX];
  uint32_t r[WORD_MAX];

  sub_ui(e, p, 1);
  pow2m(r, e, p);

  uint32_t result = r[0] - 1;

  #pragma unroll
  for(int i = 1; i < WORD_MAX; ++i)
    result |= r[i];

  return (result == 0);
}

__device__ void add_result(uint64_t *nonce_offsets, uint32_t *nonce_meta, uint32_t *nonce_count, uint64_t &offset, uint32_t &meta)
{
  atomicMin(nonce_count, OFFSETS_MAX);
  uint32_t i = atomicAdd(nonce_count, 1);
  if(i < OFFSETS_MAX)
  {
    nonce_offsets[i] = offset;
    nonce_meta[i] = meta;
  }

  
}

__global__ void fermat_kernel(uint64_t *in_nonce_offsets,
                              uint32_t *in_nonce_meta, 
                              uint32_t *in_nonce_count,
                              uint64_t *out_nonce_offsets,
                              uint32_t *out_nonce_meta,
                              uint32_t *out_nonce_count,
                              uint64_t *g_result_offsets,
                              uint32_t *g_result_meta,
                              uint32_t *g_result_count,
                              uint32_t *g_primes_checked,
                              uint32_t *g_primes_found, 
                              uint64_t nPrimorial,
                              uint32_t nTestOffsets,
                              uint32_t nTestLevels)
{
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < *in_nonce_count)
  {
    uint64_t nonce_offset = in_nonce_offsets[idx];
    uint32_t nonce_meta = in_nonce_meta[idx];

    uint32_t p[WORD_MAX];

      //decode meta_data
    uint32_t chain_offset = nonce_meta >> 16;
    uint32_t prime_gap = (nonce_meta >> 8) & 0xFF;
    uint32_t chain_length = nonce_meta & 0xFF;

    uint64_t primorial_offset = nPrimorial * nonce_offset;
    primorial_offset += c_offsets[chain_offset];

    if(c_quit == true)
      return;

    add_ui(p, c_zFirstSieveElement, primorial_offset);


    if(fermat_prime(p))
    {
      atomicAdd(g_primes_found, 1);
      ++chain_length;
      prime_gap = 0;
    }
    atomicAdd(g_primes_checked, 1);
    
    if(chain_length == nTestLevels)
    {
        //encode meta_data
      nonce_meta = 0;
      nonce_meta |= (chain_offset << 16);
      nonce_meta |= (prime_gap << 8);
      nonce_meta |= chain_length;

      add_result(g_result_offsets, g_result_meta, g_result_count, nonce_offset, nonce_meta);
    }

    ++chain_offset;

    if(chain_offset < nTestOffsets)
    {
      prime_gap += c_offsets[chain_offset] - c_offsets[chain_offset-1];

      if(chain_length > 0 && chain_length < nTestLevels && prime_gap <= 12)
      {   
          //encode meta_data
        nonce_meta = 0;
        nonce_meta |= (chain_offset << 16);
        nonce_meta |= (prime_gap << 8);
        nonce_meta |= chain_length;
  
        add_result(out_nonce_offsets, out_nonce_meta, out_nonce_count, nonce_offset, nonce_meta);
      }
    }   
	}
}

__global__ void fermat_launcher(uint64_t *g_nonce_offsets,
                                uint32_t *g_nonce_meta,
                                uint32_t *g_nonce_count,
                                uint64_t *g_result_offsets,
                                uint32_t *g_result_meta,
                                uint32_t *g_result_count,
                                uint32_t *g_primes_checked,
                                uint32_t *g_primes_found,
                                uint64_t nPrimorial,
                                uint32_t nTestOffsets,
                                uint32_t nTestLevels)
{
  dim3 block(512);
  int i = 0;
  uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  
  uint64_t *in_nonce_offsets = g_nonce_offsets;
  uint32_t *in_nonce_meta = g_nonce_meta;
  uint32_t *in_nonce_count = g_nonce_count;

  uint64_t *out_nonce_offsets = g_nonce_offsets + OFFSETS_MAX;
  uint32_t *out_nonce_meta = g_nonce_meta + OFFSETS_MAX;
  uint32_t *out_nonce_count = g_nonce_count + 1;

  //printf("nonce_count=%d\n", *in_nonce_count);

    //launch child processes to test results
  if(tid == 0 && *in_nonce_count > 0)
  {
    if(*in_nonce_count >= OFFSETS_MAX)
      printf("Offsets Max Exceeded. Choose Larger Offset Limit or Use more Sieving Primes or Offsets.\n");

    *g_result_count = 0;
    *g_primes_checked = 0;
    *g_primes_found = 0;
    
    //printf("nonce_count=%d\n", *in_nonce_count);


    while(*in_nonce_count > 0 && c_quit == false)
    {

      dim3 grid((*in_nonce_count+block.x-1)/block.x);

        //recycle results and place valid results in result queue
      *out_nonce_count = 0;

      __syncthreads();

      //printf("fermat_kernel<<<%d, %d>>>\n", grid.x, block.x);
      fermat_kernel<<<grid, block, 0>>>(in_nonce_offsets,  in_nonce_meta,  in_nonce_count,
                                        out_nonce_offsets, out_nonce_meta, out_nonce_count,
                                        g_result_offsets, g_result_meta, g_result_count,
                                        g_primes_checked, g_primes_found, nPrimorial, nTestOffsets, nTestLevels);

      cudaDeviceSynchronize();

      __syncthreads();

        //flip between working buffers
      in_nonce_offsets = g_nonce_offsets + i * OFFSETS_MAX;
      in_nonce_meta  = g_nonce_meta + i * OFFSETS_MAX;
      in_nonce_count = g_nonce_count + i;

      i ^= 1;

      out_nonce_offsets = g_nonce_offsets + i * OFFSETS_MAX;
      out_nonce_meta  = g_nonce_meta + i * OFFSETS_MAX;
      out_nonce_count = g_nonce_count + i;
    }

    *in_nonce_count = 0;
    *out_nonce_count = 0;
  }
}


extern "C" __host__ void cuda_fermat(uint32_t thr_id,
                                     uint32_t sieve_index,
                                     uint32_t test_index,
                                     uint64_t nPrimorial,
                                     uint32_t nTestLevels)
{ 
  uint32_t curr_sieve = sieve_index % FRAME_COUNT;
  uint32_t curr_test = test_index % FRAME_COUNT;

      //make sure compaction is finished before testing
  CHECK(cudaStreamWaitEvent(d_fermat_Stream[thr_id], d_compact_Event[thr_id][curr_sieve],  0));

  //printf("fermat_launcher<<<%d, %d>>>\n", 1, 1);
  fermat_launcher<<<1, 1, 0, d_fermat_Stream[thr_id]>>>(frameResources[thr_id].d_nonce_offsets[curr_test], 
                                       frameResources[thr_id].d_nonce_meta[curr_test], 
                                       frameResources[thr_id].d_nonce_count[curr_test],
                                       frameResources[thr_id].d_result_offsets[curr_test],
                                       frameResources[thr_id].d_result_meta[curr_test],
                                       frameResources[thr_id].d_result_count[curr_test],
                                       frameResources[thr_id].d_primes_checked[curr_test],
                                       frameResources[thr_id].d_primes_found[curr_test], 
                                       nPrimorial, nOffsetsTest, nTestLevels);

    //device to host pinned
  CHECK(cudaMemcpyAsync(frameResources[thr_id].h_result_offsets[curr_test], 
                        frameResources[thr_id].d_result_offsets[curr_test], 
                        OFFSETS_MAX * sizeof(uint64_t), cudaMemcpyDeviceToHost, d_fermat_Stream[thr_id]));

  CHECK(cudaMemcpyAsync(frameResources[thr_id].h_result_meta[curr_test], 
                        frameResources[thr_id].d_result_meta[curr_test], 
                        OFFSETS_MAX * sizeof(uint32_t), cudaMemcpyDeviceToHost, d_fermat_Stream[thr_id]));

  CHECK(cudaMemcpyAsync(frameResources[thr_id].h_result_count[curr_test], 
                        frameResources[thr_id].d_result_count[curr_test], 
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, d_fermat_Stream[thr_id]));

  CHECK(cudaMemcpyAsync(frameResources[thr_id].h_primes_checked[curr_test], 
                        frameResources[thr_id].d_primes_checked[curr_test], 
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, d_fermat_Stream[thr_id]));

  CHECK(cudaMemcpyAsync(frameResources[thr_id].h_primes_found[curr_test], 
                        frameResources[thr_id].d_primes_found[curr_test], 
                        sizeof(uint32_t), cudaMemcpyDeviceToHost, d_fermat_Stream[thr_id]));

  CHECK(cudaEventRecord(d_fermat_Event[thr_id][curr_test], d_fermat_Stream[thr_id]));
}

extern "C" void cuda_results(uint32_t thr_id,
                             uint32_t test_index,
                             uint64_t *result_offsets,
                             uint32_t *result_meta,
                             uint32_t *result_count,
                             uint32_t *primes_checked,
                             uint32_t *primes_found)
{
  *result_count = 0;
  *primes_checked = 0;
  *primes_found = 0;

  if(test_index < 1)
    return;

  uint32_t prev = (test_index - 1) % FRAME_COUNT; 

  if(cudaEventQuery(d_fermat_Event[thr_id][prev]) == cudaErrorNotReady)
    return;

  *primes_checked = *frameResources[thr_id].h_primes_checked[prev];
  *primes_found   = *frameResources[thr_id].h_primes_found[prev];

  *frameResources[thr_id].h_primes_checked[prev] = 0;
  *frameResources[thr_id].h_primes_found[prev] = 0;

  *result_count = *frameResources[thr_id].h_result_count[prev];
    
  if(*result_count == 0)
    return;

  memcpy(result_offsets, frameResources[thr_id].h_result_offsets[prev],
         *result_count * sizeof(uint64_t));

  memcpy(result_meta, frameResources[thr_id].h_result_meta[prev],
         *result_count * sizeof(uint32_t));

  *frameResources[thr_id].h_result_count[prev] = 0;
  *frameResources[thr_id].h_primes_checked[prev] = 0;
  *frameResources[thr_id].h_primes_found[prev] = 0;
}

extern "C" void cuda_init_counts(uint32_t thr_id)
{
  uint32_t zero[BUFFER_COUNT] = {0};

  cudaDeviceSynchronize();

  for(int i = 0; i < FRAME_COUNT; ++i)
  {
    *frameResources[thr_id].h_nonce_count[i] = 0;

    CHECK(cudaMemcpy(frameResources[thr_id].d_nonce_count[i],
                     zero,
                     sizeof(uint32_t) * BUFFER_COUNT,
                     cudaMemcpyHostToDevice));
  }
}

extern "C" void cuda_set_test_offsets(uint32_t thr_id, uint32_t *offsets, uint32_t count)
{
  nOffsetsTest = count;

  if(nOffsetsTest > 32)
    exit(1);

  CHECK(cudaMemcpyToSymbol(c_offsets, offsets, 
                           nOffsetsTest*sizeof(uint32_t), 0, cudaMemcpyHostToDevice));

}

extern "C" void cuda_init_fermat(uint32_t thr_id)
{
  CHECK(cudaStreamCreateWithFlags(&d_fermat_Stream[thr_id], cudaStreamNonBlocking));

  for(int i = 0; i < FRAME_COUNT; ++i)
  {
    CHECK(cudaEventCreateWithFlags(&d_fermat_Event[thr_id][i], cudaEventDisableTiming));
  }
}

extern "C" void cuda_free_fermat(uint32_t thr_id)
{
  CHECK(cudaStreamDestroy(d_fermat_Stream[thr_id]));

  for(int i = 0; i < FRAME_COUNT; ++i)
  {
    CHECK(cudaEventDestroy(d_fermat_Event[thr_id][i]));
  }
}

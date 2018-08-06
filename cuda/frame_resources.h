/*******************************************************************************************
   
 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php
  
*******************************************************************************************/
#ifndef FRAME_RESOURCES_H
#define FRAME_RESOURCES_H

#include <stdio.h>
#include <stdint.h>

#define CHECK(call)                                                       \
{                                                                         \
  const cudaError_t error = call;                                         \
  if(error != cudaSuccess)                                                \
  {                                                                       \
    printf("Error: %s:%d, ", __FILE__, __LINE__);                         \
    printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));   \
    exit(1);                                                              \
  }                                                                       \
}

#define OFFSETS_MAX (1 << 18)  
#define FRAME_COUNT 2
#define BUFFER_COUNT 2
#define GPU_MAX 8
struct FrameResource
{
    //testing
  uint64_t *d_result_offsets[FRAME_COUNT];
  uint64_t *h_result_offsets[FRAME_COUNT];

  uint32_t *d_result_meta[FRAME_COUNT];
  uint32_t *h_result_meta[FRAME_COUNT];

  uint32_t *d_result_count[FRAME_COUNT];
  uint32_t *h_result_count[FRAME_COUNT];

  uint32_t *d_primes_checked[FRAME_COUNT];
  uint32_t *h_primes_checked[FRAME_COUNT];

  uint32_t *d_primes_found[FRAME_COUNT];
  uint32_t *h_primes_found[FRAME_COUNT];
  
  
    //compacting
  uint64_t *d_nonce_offsets[FRAME_COUNT];
  uint32_t *d_nonce_meta[FRAME_COUNT];
  uint32_t *d_nonce_count[FRAME_COUNT];

  uint32_t *h_nonce_count[FRAME_COUNT];

   //sieving
  uint16_t *d_prime_remainders[FRAME_COUNT];
  uint8_t  *d_bit_array_sieve[FRAME_COUNT];
};

#endif

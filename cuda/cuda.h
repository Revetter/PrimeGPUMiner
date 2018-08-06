/*******************************************************************************************
   
 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php
  
*******************************************************************************************/
extern int device_map[8];

extern "C" int cuda_num_devices();

extern "C" const char* cuda_devicename(int index);

extern "C" void cuda_init(int thr_id);

extern "C" void cuda_free(int deviceCount);

extern "C" void cuda_set_zTempVar(uint32_t thr_id, const uint64_t *limbs);

extern "C" void cuda_set_primes(uint32_t thr_id, uint32_t *primes, 
                                uint32_t *inverses, uint64_t *invk, 
                                uint32_t nPrimeLimit, 
                                uint32_t nBitArray_Size, 
                                uint32_t sharedSizeKB,
                                uint32_t nPrimorialEndPrime,
                                uint32_t nPrimeLimitA);

extern "C" void cuda_compute_base_remainders(uint32_t thr_id, 
                                             uint32_t nPrimorialEndPrime, 
                                             uint32_t nPrimeLimit);

extern "C" void cuda_set_offsets(uint32_t thr_id, uint32_t *OffsetsA, uint32_t nOffsetsA,
                                                  uint32_t *OffsetsB, uint32_t nOffsetsB);

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
                                        uint32_t test_index);

extern "C" void cuda_wait_sieve(uint32_t thr_id, uint32_t sieve_index);

extern "C" void cuda_reset_device();

extern "C" void cuda_device_synchronize();

extern "C" void cuda_init_sieve(uint32_t thr_id);
extern "C" void cuda_free_sieve(uint32_t thr_id);



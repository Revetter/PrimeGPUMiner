/*******************************************************************************************
   
 Nexus Earth 2018

 [Scale Indefinitely] BlackJack. http://www.opensource.org/licenses/mit-license.php
  
*******************************************************************************************/
#ifndef FERMAT_H
#define FERMAT_H


#define WORD_MAX 32


struct uint1024;

typedef unsigned uint32_t;
typedef unsigned long uint64_t;

extern "C" void cuda_init_fermat(uint32_t thr_id);
extern "C" void cuda_free_fermat(uint32_t thr_id);

extern "C" void cuda_set_FirstSieveElement(uint32_t thr_id, 
                                           uint32_t *limbs);

extern "C" void cuda_init_counts(uint32_t thr_id);

extern "C" void cuda_set_test_offsets(uint32_t thr_id, uint32_t *offsets, uint32_t count);

extern "C" void cuda_set_quit(uint32_t quit);

extern "C" void cuda_fermat(uint32_t thr_id,
                            uint32_t sieve_index,
                            uint32_t test_index,
                            uint64_t nPrimorial,
                            uint32_t nTestLevels);

extern "C" void cuda_results(uint32_t thr_id,
                             uint32_t test_index,
                             uint64_t *result_offsets,
                             uint32_t *result_meta,
                             uint32_t *result_count,
                             uint32_t *primes_checked,
                             uint32_t *primes_found);


#endif

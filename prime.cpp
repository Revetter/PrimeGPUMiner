/*******************************************************************************************
 
			Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++
   
 [Learn, Create, but do not Forge] Viz. http://www.opensource.org/licenses/mit-license.php
 [Scale Indefinitely]        BlackJack.
  
*******************************************************************************************/

#include "core.h"
#include "cuda/cuda.h"
#include "cuda/fermat.h"
#include "cuda/frame_resources.h"

// use e.g. latest primesieve 0.5.4 package
#include <primesieve.hpp>

// used for implementing a work queue to submit work for CPU verification
#include <boost/thread/thread.hpp>

#include <queue>
#include <atomic>
#include <algorithm>
#include <inttypes.h>

using namespace std;

uint32_t *primes;
uint32_t *inverses;
uint64_t *invK;


uint32_t nBitArray_Size[GPU_MAX] = { 0 };
mpz_t  zPrimorial;

static uint64_t *static_nonce_offsets[GPU_MAX] =   {0,0,0,0,0,0,0,0};
static uint32_t *static_nonce_meta[GPU_MAX] =           {0,0,0,0,0,0,0,0};

extern uint64_t base_offset;
extern std::vector<uint32_t> offsetsTest;
extern std::vector<uint32_t> offsetsA;
extern std::vector<uint32_t> offsetsB;

const uint32_t nPrimorialEndPrime = 8;

uint32_t nPrimeLimitA[GPU_MAX] = { 0 };
uint32_t nPrimeLimitB[GPU_MAX] = { 0 };
uint32_t nSieveIterationsLog2[GPU_MAX] = { 0 };
uint32_t nTestLevels[GPU_MAX] = { 0 };
uint32_t nPrimeLimit = 0;

uint32_t nSharedSizeKB[8] = { 48 };
uint32_t nThreadsKernelA[8] = { 768 };

std::atomic<uint32_t> nFourChainsFoundCounter;
std::atomic<uint32_t> nFiveChainsFoundCounter;
std::atomic<uint32_t> nSixChainsFoundCounter;
std::atomic<uint32_t> nSevenChainsFoundCounter;
std::atomic<uint32_t> nEightChainsFoundCounter;
std::atomic<uint32_t> nNineChainsFoundCounter;

extern volatile unsigned int nBestHeight;
extern std::atomic<uint64_t> SievedBits;
extern std::atomic<uint64_t> PrimesFound;
extern std::atomic<uint64_t> PrimesChecked;
extern std::atomic<uint64_t> Tests_GPU;
extern std::atomic<uint64_t> Tests_CPU;
std::atomic<bool> quit;

uint64_t mpz2ull(mpz_t z)
{
	uint64_t result = 0;
	mpz_export(&result, 0, 0, sizeof(uint64_t), 0, 0, z);
	return result;
}
 
uint32_t *make_primes(uint32_t limit) {
	std::vector<uint32_t> primevec;
	primesieve::generate_n_primes(limit, &primevec);
	primes = (uint32_t *)malloc((limit + 1) * sizeof(uint32_t));
	primes[0] = limit;
	memcpy(&primes[1], &primevec[0], limit*sizeof(uint32_t));
	return primes;
}

#define MAX(a,b) ( (a) > (b) ? (a) : (b) )
#define MIN(a,b) ( (a) < (b) ? (a) : (b) )

namespace Core
{
	/** Divisor bit_array_sieve for Prime Searching. **/
	std::vector<uint32_t> DIVISOR_SIEVE;
	void fermat_gpu_benchmark();
	void InitializePrimes()
	{		
		printf("\nGenerating primes...\n");
		primes = make_primes(nPrimeLimit);
		printf("%d primes generated\n", primes[0]);

		mpz_init(zPrimorial);
		mpz_set_ui(zPrimorial, 1);
		double max_sieve = pow(2.0, 64);
		for (uint32_t i=1; i<nPrimorialEndPrime; ++i)
		{
			mpz_mul_ui(zPrimorial, zPrimorial, primes[i]);
			max_sieve /= primes[i];
		}
		gmp_printf("\nPrimorial: %Zd\n", zPrimorial);

		printf("Last Primorial Prime = %u\n", primes[nPrimorialEndPrime-1]);
		printf("First Sieving Prime = %u\n", primes[nPrimorialEndPrime]);

		int nSize = (int)mpz_sizeinbase(zPrimorial,2);
		printf("Primorial Size = %d-bit\n", nSize);
		printf("Max. sieve size: %" PRIu64 " bits\n", (uint64_t)max_sieve);

		inverses=(uint32_t *) malloc((nPrimeLimit+1)*sizeof(uint32_t));
		memset(inverses, 0, (nPrimeLimit+1) * sizeof(uint32_t));

		mpz_t zPrime, zInverse, zResult;

		mpz_init(zPrime);
		mpz_init(zInverse);
		mpz_init(zResult);

		printf("\nGenerating inverses...\n");

		for(uint32_t i=nPrimorialEndPrime; i<=nPrimeLimit; ++i)
		{
			mpz_set_ui(zPrime, primes[i]);

			int	inv = mpz_invert(zResult, zPrimorial, zPrime);
			if (inv <= 0)
			{
				printf("\nNo Inverse for prime %u at position %u\n\n", primes[i], i);
				exit(0);
			}
			else
			{
				inverses[i]  = (uint32_t)mpz_get_ui(zResult);
			}
		}

		printf("%d inverses generated\n\n", nPrimeLimit - nPrimorialEndPrime + 1);

		printf("\nGenerating invK...\n");
		invK = (uint64_t*)malloc((nPrimeLimit + 1) * sizeof(uint64_t));
		memset(invK, 0, (nPrimeLimit + 1) * sizeof(uint64_t));

		mpz_t n1, n2;
		mpz_init(n1);
		mpz_init(n2);

		mpz_set_ui(n1, 2);
		mpz_pow_ui(n1, n1, 64);

		for (uint32_t i = nPrimorialEndPrime; i <= nPrimeLimit; ++i)
		{
			mpz_div_ui(n2, n1, primes[i]);
			uint64_t recip = mpz2ull(n2);			
			invK[i] = recip;
		}

		mpz_clear(n1);
		mpz_clear(n2);
	}
	
	/** Convert Double to unsigned int Representative. Used for encoding / decoding prime difficulty from nBits. **/
	uint32_t SetBits(double nDiff)
	{
		uint32_t nBits = 10000000;
		nBits = (uint32_t)(nBits * nDiff);
		
		return nBits;
	}

	/** Determines the difficulty of the Given Prime Number.
		Difficulty is represented as so V.X
		V is the whole number, or Cluster Size, X is a proportion
		of Fermat Remainder from last Composite Number [0 - 1] **/
	double GetPrimeDifficulty(CBigNum prime, int checks)
	{
		CBigNum lastPrime = prime;
		CBigNum next = prime + 2;
		uint32_t clusterSize = 1;
		
		///largest prime gap in cluster can be +12
		///this was determined by previously found clusters up to 17 primes
		for( next ; next <= lastPrime + 12; next += 2)
		{
			if(PrimeCheck(next, checks))
			{
				lastPrime = next;
				++clusterSize;
			}
		}
		
		///calulate the rarety of cluster from proportion of fermat remainder of last prime + 2
		///keep fractional remainder in bounds of [0, 1]
		double fractionalRemainder = 1000000.0 / GetFractionalDifficulty(next);
		if(fractionalRemainder > 1.0 || fractionalRemainder < 0.0)
			fractionalRemainder = 0.0;
		
		return (clusterSize + fractionalRemainder);
	}

	double GetPrimeDifficulty2(CBigNum next, uint32_t clusterSize)
	{
		///calulate the rarety of cluster from proportion of fermat remainder of last prime + 2
		///keep fractional remainder in bounds of [0, 1]
		double fractionalRemainder = 1000000.0 / GetFractionalDifficulty(next);
		if(fractionalRemainder > 1.0 || fractionalRemainder < 0.0)
			fractionalRemainder = 0.0;
		
		return (clusterSize + fractionalRemainder);
	}

	/** Gets the unsigned int representative of a decimal prime difficulty **/
	uint32_t GetPrimeBits(CBigNum prime, int checks)
	{
		return SetBits(GetPrimeDifficulty(prime, checks));
	}

	/** Breaks the remainder of last composite in Prime Cluster into an integer. 
		Larger numbers are more rare to find, so a proportion can be determined 
		to give decimal difficulty between whole number increases. **/
	uint32_t GetFractionalDifficulty(CBigNum composite)
	{
		/** Break the remainder of Fermat test to calculate fractional difficulty [Thanks Sunny] **/
		return ((composite - FermatTest(composite, 2) << 24) / composite).getuint();
	}
	
	
	/** bit_array_sieve of Eratosthenes for Divisor Tests. Used for Searching Primes. **/
	std::vector<uint32_t> Eratosthenes(int nSieveSize)
	{
		bool *TABLE = new bool[nSieveSize];
		
		for(int nIndex = 0; nIndex < nSieveSize; nIndex++)
			TABLE[nIndex] = false;
			
			
		for(int nIndex = 2; nIndex < nSieveSize; nIndex++)
			for(int nComposite = 2; (nComposite * nIndex) < nSieveSize; nComposite++)
				TABLE[nComposite * nIndex] = true;
		
		
		std::vector<uint32_t> PRIMES;
		for(int nIndex = 2; nIndex < nSieveSize; nIndex++)
			if(!TABLE[nIndex])
				PRIMES.push_back(nIndex);

		
		printf("bit_array_sieve of Eratosthenes Generated %u Primes.\n", (uint32_t)PRIMES.size());
		
		delete[] TABLE;
		return PRIMES;
	}

	/** Determines if given number is Prime. Accuracy can be determined by "checks". 
		The default checks the Nexus Network uses is 2 **/
	bool PrimeCheck(CBigNum test, int checks)
	{
		/** Check C: Fermat Tests */
		CBigNum n = 3;
		if(FermatTest(test, n) != 1)
				return false;
		
		return true;
	}

	/** Simple Modular Exponential Equation a^(n - 1) % n == 1 or notated in Modular Arithmetic a^(n - 1) = 1 [mod n]. 
		a = Base or 2... 2 + checks, n is the Prime Test. Used after Miller-Rabin and Divisor tests to verify primality. **/
	CBigNum FermatTest(CBigNum n, CBigNum a)
	{
		CAutoBN_CTX pctx;
		CBigNum e = n - 1;
		CBigNum r;
		BN_mod_exp(&r, &a, &e, &n, pctx);
		
		return r;
	}

	/** Miller-Rabin Primality Test from the OpenSSL BN Library. **/
	bool Miller_Rabin(CBigNum n, int checks)
	{
		return (BN_is_prime(&n, checks, NULL, NULL, NULL) == 1);
	}

	uint32_t mpi_mod_int(mpz_t A, uint32_t B)
	{
		if (B == 1)
			return 0;
		else if (B == 2)
			return A[0]._mp_d[0]&1;

		#define biH (sizeof(mp_limb_t)<<2)
		int i;
		mp_limb_t b=B,x,y,z;

		for( i = A[0]._mp_alloc - 1, y = 0; i > 0; --i )
		{
			x  = A[0]._mp_d[i - 1];
			y  = ( y << biH ) | ( x >> biH );
			z  = y / b;
			y -= z * b;

			x <<= biH;
			y  = ( y << biH ) | ( x >> biH );
			z  = y / b;
			y -= z * b;
		}

		return (uint32_t)y;
	}

	static int Convert_BIGNUM_to_mpz_t(const BIGNUM *bn, mpz_t g)
	{
		bn_check_top(bn);
		if(((sizeof(bn->d[0]) * 8) == GMP_NUMB_BITS) &&
				(BN_BITS2 == GMP_NUMB_BITS)) 
		{
			/* The common case */
			if(!_mpz_realloc (g, bn->top))
				return 0;
			memcpy(&g->_mp_d[0], &bn->d[0], bn->top * sizeof(bn->d[0]));
			g->_mp_size = bn->top;
			if(bn->neg)
				g->_mp_size = -g->_mp_size;
			return 1;
		}
		else
		{
			char *tmpchar = BN_bn2hex(bn);
			if(!tmpchar) return 0;
			OPENSSL_free(tmpchar);
			return 0;
		}
	}

	boost::mutex work_mutex;
	std::deque<work_info> work_queue;
	std::queue<work_info> result_queue;	

	int scan_offsets(work_info &work)
	{
		int scanned = 0;

		work.nNonce = false;
		work.nNonceDifficulty = 0;

		mpz_t zTempVar, zN, zFirstSieveElement, zPrimeOrigin, zPrimeOriginOffset, zResidue, zTwo;
		mpz_init(zTempVar);
		mpz_init(zN);
		mpz_init_set(zFirstSieveElement, work.zFirstSieveElement.__get_mp());
		mpz_init(zPrimeOrigin);
		Convert_BIGNUM_to_mpz_t(&work.BaseHash, zPrimeOrigin);
		mpz_init(zPrimeOriginOffset);
		mpz_init(zResidue);
		mpz_init_set_ui(zTwo, 2);

		mpz_mod(zTempVar, zFirstSieveElement, zPrimorial);
		uint64_t constellation_origin = mpz_get_ui(zTempVar);

		uint64_t nNonce = 0;
		uint32_t nSieveDifficulty = 0;
		uint64_t nStart = 0;
		uint64_t nStop = 0;
		uint64_t offset = work.nonce_offset;

    uint32_t prime_gap = (work.nonce_meta >> 8) & 0xFF;
    uint32_t chain_offset = work.nonce_meta >> 16;
    uint32_t chain_length = work.nonce_meta & 0xFF;

    uint32_t thr_id = work.gpu_thread;

    //if(chain_offset == 0)
    //  printf("chain offset is 0, may be counting chain lengths twice!\n");

    if(work.nHeight != nBestHeight || nBestHeight == 0 || quit.load())
      return scanned;
    
    mpz_mul_ui(zTempVar, zPrimorial, offset);
		mpz_add(zTempVar, zFirstSieveElement, zTempVar);
    mpz_set(zPrimeOriginOffset, zTempVar);
    
    chain_offset = offsetsTest[chain_offset] + 2;
    mpz_add_ui(zTempVar, zTempVar, chain_offset);

    for(; nStart<=nStop+12; nStart += 2)
    {
      mpz_sub_ui(zN, zTempVar, 1);
      mpz_powm(zResidue, zTwo, zN, zTempVar);
      if(mpz_cmp_ui(zResidue, 1) == 0)
      {
        ++PrimesFound;
        ++chain_length;

        nStop = nStart;
      }
      ++PrimesChecked;
      ++Tests_CPU;

      mpz_add_ui(zTempVar, zTempVar, 2);
      chain_offset += 2;
    }

    if (chain_length >= 4)
	  {
	    ++scanned;
  
      mpz_sub(zTempVar, zPrimeOriginOffset, zPrimeOrigin);
			nNonce = mpz_get_ui(zTempVar);
			nSieveDifficulty = SetBits(GetPrimeDifficulty2(work.BaseHash + nNonce + chain_offset, chain_length));
      //printf("sieve difficulty: %d\n", nSieveDifficulty);

			if (nSieveDifficulty >= 40000000)
				++nFourChainsFoundCounter;
			if (nSieveDifficulty >= 50000000)
				++nFiveChainsFoundCounter;
			if (nSieveDifficulty >= 60000000)
			  ++nSixChainsFoundCounter;
	    if (nSieveDifficulty >= 70000000)
				++nSevenChainsFoundCounter;
		  if (nSieveDifficulty >= 80000000)
		    ++nEightChainsFoundCounter;
      if (nSieveDifficulty >= 90000000)
        ++nNineChainsFoundCounter;

		  if (nSieveDifficulty >= 60000000)
      {
			  printf("\n  %d-Chain Found: %f  Nonce: %016lX  %s[%d]\n\n", 
            (int)chain_length, 
            (double)nSieveDifficulty / 1e7, 
            nNonce,
            cuda_devicename(thr_id),
            thr_id);
      }

			if(nSieveDifficulty >= work.nDifficulty)
			{
				work.nNonce = nNonce;
	  	  work.nNonceDifficulty = nSieveDifficulty;
		  }
    }
 
#if TIMING
		QueryPerformanceCounter(&work.EndingTime);
#endif

		mpz_clear(zPrimeOrigin);
		mpz_clear(zPrimeOriginOffset);
		mpz_clear(zFirstSieveElement);
		mpz_clear(zResidue);
		mpz_clear(zTwo);
		mpz_clear(zN);
		mpz_clear(zTempVar);

		return scanned;
	}

	bool PrimeQuery()
	{
		work_info work;
		bool have_work = false;
		{
			boost::mutex::scoped_lock lock(work_mutex);
			if (!work_queue.empty())
			{
				work = work_queue.front();
				work_queue.pop_front();
				have_work = true;
			}
		}

		if (have_work)
		{
			int scanned = scan_offsets(work);

			if (work.nNonce != 0 && work.nNonceDifficulty > work.nDifficulty)
			{
				boost::mutex::scoped_lock lock(work_mutex);
				result_queue.emplace(work);
			}
		}
		return have_work;
	}

  void PrimeInit(int threadIndex)
  {
    cuda_init(threadIndex);
    cuda_init_sieve(threadIndex);
    cuda_init_fermat(threadIndex);

		printf("Thread %d starting up...\n", threadIndex);

    uint32_t bit_array_size = nBitArray_Size[threadIndex];

		cuda_set_primes(threadIndex, primes, inverses, invK, nPrimeLimit, 
                    bit_array_size, nSharedSizeKB[threadIndex], 
                    nPrimorialEndPrime, nPrimeLimitA[threadIndex]);

    cuda_set_offsets(threadIndex, &offsetsA[0], offsetsA.size(), 
                                  &offsetsB[0], offsetsB.size());

    cuda_set_test_offsets(threadIndex, &offsetsTest[0], offsetsTest.size());

    static_nonce_offsets[threadIndex] = (uint64_t*)malloc(OFFSETS_MAX * sizeof(uint64_t));
    static_nonce_meta[threadIndex]    = (uint32_t*)malloc(OFFSETS_MAX * sizeof(uint32_t));

  }

  void PrimeFree(int threadIndex)
  {
    delete[] static_nonce_offsets[threadIndex];
    delete[] static_nonce_meta[threadIndex];

  
    cuda_free_sieve(threadIndex);
    cuda_free_fermat(threadIndex);
  }

	void PrimeSieve(uint32_t threadIndex, uint32_t threadCount, CBigNum BaseHash, uint32_t nDifficulty, uint32_t nHeight, uint512 merkleRoot)
	{
    std::vector<uint32_t> limbs(WORD_MAX);
    size_t size = 0; 
		uint64_t result = false;
		mpz_t zPrimeOrigin;
    mpz_t zFirstSieveElement;
    mpz_t zPrimorialMod;
    mpz_t zTempVar;

    uint32_t primeLimitB = nPrimeLimitB[threadIndex];
    uint32_t primeLimitA = nPrimeLimitA[threadIndex];

    uint32_t shared_kb = nSharedSizeKB[threadIndex];

    uint32_t  bit_array_size = nBitArray_Size[threadIndex];

    uint64_t *nonce_offsets = static_nonce_offsets[threadIndex];
    uint32_t *nonce_meta    = static_nonce_meta[threadIndex];

		uint32_t i = 0;
		uint32_t j = 0;
		uint32_t nSize = 0;

		mpz_init(zFirstSieveElement);
		mpz_init(zPrimorialMod);
		mpz_init(zTempVar);
		mpz_init(zPrimeOrigin);

		Convert_BIGNUM_to_mpz_t(&BaseHash, zPrimeOrigin);
    //uint32_t bits = mpz_sizeinbase(zPrimeOrigin, 2);
    //size = (bits + 31) / 32; 
    //mpz_export(&limbs[0], &size, -1, sizeof(uint32_t), 0, 0, zPrimeOrigin);
    
    cuda_init_counts(threadIndex);


		uint64_t primorial = mpz_get_ui(zPrimorial);

	  mpz_mod(zPrimorialMod, zPrimeOrigin, zPrimorial);
    mpz_sub(zPrimorialMod, zPrimorial, zPrimorialMod);
    mpz_add(zTempVar, zPrimeOrigin, zPrimorialMod);

      //compute base remainders
    cuda_set_zTempVar(threadIndex, (const uint64_t*)zTempVar[0]._mp_d);
    cuda_compute_base_remainders(threadIndex, 
                                 nPrimorialEndPrime, 
                                 nPrimeLimit);


      //compute first sieving element
    mpz_add_ui(zFirstSieveElement, zTempVar, base_offset);

      //export firstsieve element to GPU
    size = (mpz_sizeinbase(zFirstSieveElement,2) + 31) / 32;    
    mpz_export(&limbs[0], &size, -1, sizeof(uint32_t), 0, 0, zFirstSieveElement);

	  cuda_set_FirstSieveElement(threadIndex, &limbs[0]);
    cuda_set_quit(0);
    
    uint32_t count = 0;
    uint32_t primes_checked = 0;
    uint32_t primes_found = 0;
    uint32_t sieve_index = 0;
    uint32_t test_index = 0;
    bool sieve_finished = false;

    uint32_t test_levels = nTestLevels[threadIndex];

    uint32_t nIterations = 1 << nSieveIterationsLog2[threadIndex];
   
    while(nHeight && nHeight == nBestHeight && quit.load() == false)
    {
        //sieve bit array and compact test candidate nonces
	    sieve_finished = cuda_compute_primesieve(threadIndex,
                                               threadCount,
                                               shared_kb,
                                               nThreadsKernelA[threadIndex], 
                                               base_offset,
                                               primorial, 
                                               nPrimorialEndPrime,  
                                               primeLimitA, 
                                               primeLimitB,
				                                       bit_array_size, 
                                               nDifficulty,
                                               sieve_index,
                                               test_index);

        //after the number of iterations have been satisfied, start filling next queue
      
      if(sieve_index > 0 && sieve_index % nIterations == 0 && sieve_finished)
      {
          //test results
        cuda_fermat(threadIndex, sieve_index, test_index, primorial, test_levels);
        ++test_index;
      }

        //obtain the final results and push them onto the queue
      cuda_results(threadIndex, test_index, nonce_offsets, nonce_meta,
       &count, &primes_checked, &primes_found);

      PrimesChecked += primes_checked;
      Tests_GPU += primes_checked;
      PrimesFound += primes_found;

      if(nHeight != nBestHeight || quit.load())
        break;
      
      if(count)
      {
        //uint16_t lengths[8] = {0,0,0,0,0,0,0,0};
        for(int i = 0; i < count; ++i)
        {
          uint32_t chain_offset = nonce_meta[i] >> 16;
          uint32_t prime_gap = (nonce_meta[i] >> 8) & 0xFF;
          uint32_t chain_length = nonce_meta[i] & 0xFF;


          //++lengths[chain_length];

 	        boost::mutex::scoped_lock lock(work_mutex);
  	      work_queue.emplace_back(
            work_info(BaseHash, 
              nDifficulty, 
              nHeight, 
              threadIndex, 
              nonce_offsets[i], 
              nonce_meta[i],
              zFirstSieveElement, 
              primeLimitB,
              merkleRoot));
        }

        //printf("%d:\t", threadIndex);
        //for(int i = 1; i < 8; ++i)
        //  if(lengths[i])
        //    printf("%d\t", lengths[i]);
        //printf("\n");

        count = 0;
      }

      if(sieve_finished)
      {
        ++sieve_index;
        SievedBits += bit_array_size;
      }
      //else
      //{
      //    /*  wait for current sieve to finish and free up CPU 
      //        for other useful threaded tasks */
      //  cuda_wait_sieve(threadIndex, sieve_index);
      //}

        /*change frequency of looping for better GPU utilization, can lead to
          lower latency from calling thread waking a blocking-sync thread */ 
      Sleep(1); 
    }

    cuda_set_quit(1);

		mpz_clear(zPrimeOrigin);
		mpz_clear(zFirstSieveElement);
		mpz_clear(zPrimorialMod);
		mpz_clear(zTempVar);
  }
}


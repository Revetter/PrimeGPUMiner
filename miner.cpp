/*******************************************************************************************
 
			Hash(BEGIN(Satoshi[2010]), END(Sunny[2012])) == Videlicet[2014] ++
   
 [Learn, Create, but do not Forge] Viz. http://www.opensource.org/licenses/mit-license.php
 [Scale Indefinitely]        BlackJack.
  
*******************************************************************************************/
#include "core.h"
#include "cuda/cuda.h"
#include "iniParser.h"

#include <inttypes.h>
#include <queue>
#include <fstream>
#include <sstream>
#include <thread>

#include "cuda/frame_resources.h"

// fix  unresolved external symbol __iob_func error when linking OpenSSL 1.0.2j to this miner
// see http://stackoverflow.com/questions/30412951/unresolved-external-symbol-imp-fprintf-and-imp-iob-func-sdl2
#ifdef _WIN32
FILE _iob[] = { *stdin, *stdout, *stderr };
extern "C" FILE * __cdecl __iob_func(void) { return _iob; }
#endif

volatile unsigned int nBlocksFoundCounter = 0;
volatile unsigned int nBlocksAccepted = 0;
volatile unsigned int nBlocksRejected = 0;
volatile unsigned int nDifficulty = 0;
volatile unsigned int nBestHeight = 0;
extern volatile unsigned int nInternalPrimeLimit;
volatile bool isBlockSubmission = false;
unsigned int nStartTimer = 0;
std::atomic<uint64_t> SievedBits;
std::atomic<uint64_t> Tests_CPU;
std::atomic<uint64_t> Tests_GPU;
std::atomic<uint64_t> PrimesFound;
std::atomic<uint64_t> PrimesChecked;

extern uint32_t nBitArray_Size[GPU_MAX];
extern uint32_t nPrimeLimitA[GPU_MAX];
extern uint32_t nPrimeLimitB[GPU_MAX];
extern uint32_t nPrimeLimit;           // will be set to the maximum of any in nPrimeLimitB[]
extern uint32_t nSharedSizeKB[GPU_MAX];
extern uint32_t nThreadsKernelA[GPU_MAX];
extern uint32_t nTestLevels[GPU_MAX];
extern uint32_t nSieveIterationsLog2[GPU_MAX];

extern std::atomic<uint32_t> nFourChainsFoundCounter;
extern std::atomic<uint32_t> nFiveChainsFoundCounter;
extern std::atomic<uint32_t> nSixChainsFoundCounter;
extern std::atomic<uint32_t> nSevenChainsFoundCounter;
extern std::atomic<uint32_t> nEightChainsFoundCounter;
extern std::atomic<uint32_t> nNineChainsFoundCounter;

extern std::atomic<bool> quit;

uint64_t base_offset = 0;
std::vector<uint32_t> offsetsTest;
std::vector<uint32_t> offsetsA;
std::vector<uint32_t> offsetsB;

#if WIN32
static inline void affine_to_cpu(int id, int cpu)
{
    DWORD mask = 1 << cpu;
    SetThreadAffinityMask(GetCurrentThread(), mask);
}
#else
static inline void affine_to_cpu(int id, int cpu)
{
	cpu_set_t set;

	CPU_ZERO(&set);
	CPU_SET(cpu, &set);
	sched_setaffinity(0, sizeof(&set), &set);
}
#endif

void signal_handler(const boost::system::error_code& error,
	int signal_number)
{
}

namespace Core
{

	/** Class to hold the basic data a Miner will use to build a Block.
		Used to allow one Connection for any amount of threads. **/

	extern std::queue<work_info> result_queue;
	extern std::deque<work_info> work_queue;
	extern boost::mutex work_mutex;

	class MinerThreadGPU
	{
	public:
		uint32_t threadIndex;
    uint32_t threadCount;
		int threadAffinity;
		CBlock* BLOCK;
		bool fBlockFound, fNewBlock;
		LLP::Thread_t THREAD;
		boost::mutex MUTEX;
		uint32_t nSearches;
    bool fReady;
		
		MinerThreadGPU(uint32_t tid, uint32_t tcount, int affinity) 
    : threadIndex(tid)
    , threadCount(tcount)
    , threadAffinity(affinity)
    , BLOCK(NULL) 
    , fBlockFound(false)
    , fNewBlock(true)
    , nSearches(0)
    , fReady(false)
    , THREAD(boost::bind(&MinerThreadGPU::PrimeMiner, this)) { }
		
		
		/** Main Miner Thread. Bound to the class with boost. Might take some rearranging to get working with OpenCL. **/
		void PrimeMiner()
		{
			affine_to_cpu(threadIndex, threadAffinity); // all CUDA threads run on CPU core 0 + threadIndex

      PrimeInit(threadIndex);
      fReady = true;
		
			loop
			{
				try
				{
					/* Keep thread at idle CPU usage if waiting to submit or recieve block. **/
					Sleep(1);

					if(!(fNewBlock || fBlockFound || !BLOCK))
					{
						nDifficulty = BLOCK->nBits;
						BLOCK->nNonce = 0;
						PrimeSieve(threadIndex, 
                       threadCount, 
                       BLOCK->GetPrime(), 
                       BLOCK->nBits, 
                       BLOCK->nHeight, 
                       BLOCK->hashMerkleRoot);
						
            fNewBlock = true;
					}


				}
				catch(std::exception& e){ printf("ERROR: %s\n", e.what()); }
			}
      PrimeFree(threadIndex);
		}

	};

  	class MinerThreadCPU
	{
	public:
		int threadIndex;
		int threadAffinity;
		bool fBlockFound, fNewBlock;
		LLP::Thread_t THREAD;
		boost::mutex MUTEX;

		unsigned int nSearches;
		
		MinerThreadCPU(int tid, int affinity) 
    : threadIndex(tid)
    , threadAffinity(affinity)
    , fBlockFound(false)
    , fNewBlock(true)
    , nSearches(0)
    , THREAD(boost::bind(&MinerThreadCPU::PrimeMiner, this)) 
    { 
    }
		
		
		/** Main Miner Thread. Bound to the class with boost. Might take some rearranging to get working with OpenCL. **/
		void PrimeMiner()
		{
			affine_to_cpu(threadIndex, threadAffinity);
			
			loop
			{
				try
				{
					if (!PrimeQuery())
					{
						Sleep(100);
					}

          if(quit.load())
            break;
				}
				catch(std::exception& e){ printf("ERROR: %s\n", e.what()); }
			}
		}
	};
	
	
	/** Class to handle all the Connections via Mining LLP.
		Independent of Mining Threads for Higher Efficiency. **/
	class ServerConnection
	{
	public:
		LLP::Miner* CLIENT;
		uint32_t nThreadsGPU;
    uint32_t nThreadsCPU;
    uint32_t nTimeout;
		std::vector<MinerThreadGPU*> THREADS_GPU;
    std::vector<MinerThreadCPU*> THREADS_CPU;
		LLP::Thread_t THREAD;
		LLP::Timer    TIMER;
		std::string   IP, PORT;
    bool fNewBlock;
		
		ServerConnection(std::string ip, std::string port, int nMaxThreadsGPU, int nMaxThreadsCPU, int nMaxTimeout) 
    : IP(ip)
    , PORT(port)
    , TIMER()
    , nThreadsGPU(nMaxThreadsGPU)
    , nThreadsCPU(nMaxThreadsCPU)
    , nTimeout(nMaxTimeout)
    , THREAD(boost::bind(&ServerConnection::ServerThread, this))
    , fNewBlock(true)
		{

			int affinity = 0;
			int nthr = std::thread::hardware_concurrency();

			for(uint32_t nIndex = 0; nIndex < nThreadsGPU; ++nIndex)
				THREADS_GPU.push_back(new MinerThreadGPU(nIndex, nThreadsGPU, (affinity++)%nthr));

      for(uint32_t nIndex = 0; nIndex < nThreadsCPU; ++nIndex)
      THREADS_CPU.push_back(new MinerThreadCPU(0, (affinity++)%nthr));

		}
		
		/** Reset the block on each of the Threads. **/
		void ResetThreads()
		{
			/** Reset each individual flag to tell threads to stop mining. **/
			for(int nIndex = 0; nIndex < THREADS_GPU.size(); ++nIndex)
				THREADS_GPU[nIndex]->fNewBlock   = true;

      fNewBlock = true;	
		}
		
		/** Main Connection Thread. Handles all the networking to allow
			Mining threads the most performance. **/
		void ServerThread()
		{
			/** Don't begin until all mining threads are Created. **/
			while(THREADS_GPU.size() != nThreadsGPU)
				Sleep(1);

      uint32_t ready = 0;
      while(ready != nThreadsGPU)
      {
         ready = 0;
         for(int i = 0; i < nThreadsGPU; ++i)
         {
           if(THREADS_GPU[i]->fReady)
            ++ready;
         }
         Sleep(1);
      }

				
				
			/** Initialize the Server Connection. **/
			CLIENT = new LLP::Miner(IP, PORT);
				
				
			/** Initialize a Timer for the Hash Meter. **/
			TIMER.Start();
			
			loop
			{
				try
				{
					/** Run this thread at 1 Cycle per Second. **/
					Sleep(1000);

          if(quit.load())
            break;
					
					
					/** Attempt with best efforts to keep the Connection Alive. **/
					if(!CLIENT->Connected() || CLIENT->Errors())
					{
						ResetThreads();
						
						if(!CLIENT->Connect())
							continue;
						else
							CLIENT->SetChannel(1);
					}
					
					
					/** Check the Block Height. **/
					unsigned int nHeight = CLIENT->GetHeight(nTimeout);
					if(nHeight == 0)
					{
						printf("Failed to Update Height...\n");
						CLIENT->Disconnect();
						continue;
					}
					
					/** If there is a new block, Flag the Threads to Stop Mining. **/
					if(nHeight != nBestHeight)
					{
						isBlockSubmission = false;
						nBestHeight = nHeight;
						printf("\n[MASTER] Nexus Network: New Block %u\n", nHeight);

						ResetThreads();
					}

					/** Rudimentary Meter **/
					if(TIMER.Elapsed() >= 15)
					{
						time_t now = time(0);
						uint32_t SecondsElapsed = (uint32_t)now - nStartTimer;
						uint32_t nElapsed = TIMER.Elapsed();
							
						uint64_t gibps = SievedBits.load() / nElapsed;
						SievedBits = 0;

            uint64_t tests_cpu = Tests_CPU.load();
            uint64_t tests_gpu = Tests_GPU.load();
            //uint64_t tests_total = tests_cpu + tests_gpu;

						uint64_t tps_cpu = tests_cpu / nElapsed;
            uint64_t tps_gpu = tests_gpu / nElapsed;
            //uint64_t tps_total = tests_total / nElapsed;

            //double percent_gpu = (double)(100 * tests_gpu) / tests_total;
            //double percent_cpu = (double)(100 * tests_cpu) / tests_total;

            Tests_CPU = 0;
            Tests_GPU = 0;

            uint64_t checked = PrimesChecked.load();
            uint64_t found = PrimesFound.load();
            
						double pratio = 0.0;

            if(checked)
              pratio = (double)(100 * found) / checked;

						printf("\n[METERS] %u Block(s) A=%u R=%u | Height = %u | Diff = %f | %02d:%02d:%02d\n", 
              nBlocksFoundCounter,
              nBlocksAccepted,
              nBlocksRejected,
              nBestHeight, 
              (double)nDifficulty/10000000.0, 
              (SecondsElapsed/3600)%60, 
              (SecondsElapsed/60)%60, 
              (SecondsElapsed)%60);

						printf("[METERS] Sieved %5.2f GiB/s | Tested %lu T/s GPU, %lu T/s CPU | Ratio: %.3f %%\n", 
              (double)gibps / (1 << 30), 
              tps_gpu,
              tps_cpu, 
              pratio);

						printf("[METERS] Clusters Found: Four=%u | Five=%u | Six=%u | Seven=%u | Eight=%u | Nine=%u\n", 
              nFourChainsFoundCounter.load(), 
              nFiveChainsFoundCounter.load(), 
              nSixChainsFoundCounter.load(), 
              nSevenChainsFoundCounter.load(),
              nEightChainsFoundCounter.load(),
              nNineChainsFoundCounter.load());
						
						TIMER.Reset();
					}

          	
            /** Attempt to get a new block from the Server if Thread needs One. **/
          if(fNewBlock)
          {
              /** Retrieve new block from Server. **/
					  CBlock* BLOCK = CLIENT->GetBlock(nTimeout);

              /** If the Block didn't come in properly, Reconnect to the Server. **/
					  if(!BLOCK)
						  CLIENT->Disconnect();

            for(int nIndex = 0; nIndex < THREADS_GPU.size(); ++nIndex)
					  {
							  /** If the block is good, tell the Mining Thread its okay to Mine. **/
						  THREADS_GPU[nIndex]->BLOCK = BLOCK;							
				      THREADS_GPU[nIndex]->fBlockFound = false;
						  THREADS_GPU[nIndex]->fNewBlock   = false;
					  }

            fNewBlock = false;
          }

					{
						boost::mutex::scoped_lock lock(work_mutex);
						while(result_queue.empty() == false)
						{
							++nBlocksFoundCounter;

							printf("\nSubmitting Block...\n");
							work_info work = result_queue.front();
							result_queue.pop();
							
							double difficulty = (double)work.nNonceDifficulty / 10000000.0;

							printf("\n[MASTER] Prime Cluster of Difficulty %f Found\n", difficulty);
								
							/** Attempt to Submit the Block to Network. **/
							unsigned char RESPONSE = CLIENT->SubmitBlock(work.merkleRoot, work.nNonce, nTimeout);
							
							/** Check the Response from the Server.**/
							if(RESPONSE == 200)
							{
								printf("\n[MASTER] Block Accepted By Nexus Network.\n");
								
								ResetThreads();
								++nBlocksAccepted;
							}
							else if(RESPONSE == 201)
							{
								printf("\n[MASTER] Block Rejected by Nexus Network.\n");
								isBlockSubmission = false;
								++nBlocksRejected;
							}
								
							/** If the Response was Bad, Reconnect to Server. **/
							else 
							{
								printf("\n[MASTER] Failure to Submit Block. Reconnecting...\n");
								CLIENT->Disconnect();
								break;
							}
						}
					}
				}
				catch(std::exception& e)
				{
					printf("%s\n", e.what()); CLIENT = new LLP::Miner(IP, PORT); 
				}
			}
		}
	};
}

void load_offsets()
{
        //get offsets used for sieving from file
    std::ifstream fin("offsets.ini");
    if(!fin.is_open())
    {
      printf("Error! could not find offsets.ini!\n");
      exit(1);
    }
    std::string O, T, A, B;
    std::getline(fin, O, '#');

    std::getline(fin, T);
    std::getline(fin, T, '#');

    std::getline(fin, A);
    std::getline(fin, A, '#');

    std::getline(fin, B);
    std::getline(fin, B, '#');
    fin.close();
    
    std::stringstream sO(O);
    std::stringstream sT(T);
    std::stringstream sA(A);
    std::stringstream sB(B);
    uint32_t o;

    sO >> base_offset;
    
    while (sT >> o)
    {
      offsetsTest.push_back(o);
      if(sT.peek() == ',')
        sT.ignore();
    }
    while (sA >> o)
    {
      offsetsA.push_back(o);
      if(sA.peek() == ',')
        sA.ignore();
    }
    while (sB >> o)
    {
      offsetsB.push_back(o);
      if(sB.peek() == ',')
        sB.ignore();
    }

    printf("\nbase_offset = %lu\n", base_offset);
    printf("offsetsTest = %u", offsetsTest[0]);
    for(int i = 1; i < offsetsTest.size(); ++i)
      printf(", %u", offsetsTest[i]);
    printf("\noffsetsA    = %u", offsetsA[0]);
    for(int i = 1; i < offsetsA.size(); ++i)
      printf(", %u", offsetsA[i]);
    printf("\noffsetsB    = %u", offsetsB[0]);
    for(int i = 1; i < offsetsB.size(); ++i)
      printf(", %u", offsetsB[i]);
    printf("\n");
}

#include <primesieve.hpp>


int main(int argc, char *argv[])
{
	if(argc < 3)
	{
		printf("Too Few Arguments. The Required Arguments are Ip and Port\n");
		printf("Default Arguments are Total Threads = nVidia GPUs and Connection Timeout = 10 Seconds\n");
		printf("Format for Arguments is 'IP PORT DEVICELIST CPUTHREADS TIMEOUT'\n");
		
		Sleep(10000);
		
		return 0;
	}

	// the io_service.run() replaces the nonterminating sleep loop
	boost::asio::io_service io_service;

	// construct a signal set registered for process termination.
	boost::asio::signal_set signals(io_service, SIGINT, SIGTERM);

	// start an asynchronous wait for one of the signals to occur.
	signals.async_wait(signal_handler);

	std::string IP = argv[1];
	std::string PORT = argv[2];
	unsigned int nThreadsGPU = GetTotalCores();
  unsigned int nThreadsCPU = 6;
	unsigned int nTimeout = 10;
	
	if(argc > 3) {
		int num_processors = nThreadsGPU;
		char * pch = strtok (argv[3],",");
		nThreadsGPU = 0;
		while (pch != NULL) {
			if (pch[0] >= '0' && pch[0] <= '9' && pch[1] == '\0')
			{
				if (atoi(pch) < num_processors)
					device_map[nThreadsGPU++] = atoi(pch);
				else 
        {
					fprintf(stderr, "Non-existant CUDA device #%d specified\n", atoi(pch));
					exit(1);
				}
			} 
      else 
      {

				fprintf(stderr, "Non-existant CUDA device '%s' specified\n", pch);
			  exit(1);
			}
			pch = strtok (NULL, ",");
		}
	}
	
	if(argc > 4)
		nTimeout = boost::lexical_cast<int>(argv[4]);
	
	printf("\nLoading configuration...\n");
	
  std::ifstream t("config.ini");
  std::stringstream buffer;
  buffer << t.rdbuf();
	std::string config = buffer.str();

	CIniParser parser;
	if (parser.Parse(config.c_str()) == false)
	{
		fprintf(stderr, "Unable to parse config.ini");
	}

	for (int i=0 ; i < nThreadsGPU; ++i)
	{
		const char *devicename = cuda_devicename(device_map[i]);

		if (!parser.GetValueAsInteger(devicename, "nPrimeLimitB", (int*)&nPrimeLimitB[i]))
			parser.GetValueAsInteger("GENERAL", "nPrimeLimitB", (int*)&nPrimeLimitB[i]);
		if (!parser.GetValueAsInteger(devicename, "nBitArray_Size", (int*)&nBitArray_Size[i]))
			parser.GetValueAsInteger("GENERAL", "nBitArray_Size", (int*)&nBitArray_Size[i]);
    if (!parser.GetValueAsInteger(devicename, "nSieveIterationsLog2", (int*)&nSieveIterationsLog2[i]))
			parser.GetValueAsInteger("GENERAL", "nSieveIterationsLog2", (int*)&nSieveIterationsLog2[i]);

    if (!parser.GetValueAsInteger(devicename, "nTestLevels", (int*)&nTestLevels[i]))
			parser.GetValueAsInteger("GENERAL", "nTestLevels", (int*)&nTestLevels[i]);

    nPrimeLimitA[i] = 512;
    nSharedSizeKB[i] = 32;
    nThreadsKernelA[i] = 512;

		printf("\nGPU thread %d, device %d [%s]\n", i, device_map[i], devicename);
		//printf("nPrimeLimitA = %d\n", nPrimeLimitA[i]);
		printf("nPrimeLimitB = %d\n", nPrimeLimitB[i]);
		printf("nBitArray_Size = %d\n", nBitArray_Size[i]);
		printf("nSieveIterations = %d\n", (1 << nSieveIterationsLog2[i]));
    printf("nTestLevels = %d\n", nTestLevels[i]);

		if (nPrimeLimitB[i] > nPrimeLimit)
      nPrimeLimit = nPrimeLimitB[i];
	}

  quit.store(false);

  load_offsets();

	Core::InitializePrimes();

	nStartTimer = (unsigned int)time(0);
	printf("Initializing Miner %s:%s Threads = %i, Timeout = %i\n", 
    IP.c_str(), PORT.c_str(), nThreadsGPU, nTimeout);

	Core::ServerConnection MINERS(IP, PORT, nThreadsGPU, nThreadsCPU, nTimeout);

	io_service.run();

  quit.store(true);
  

  cuda_free(nThreadsGPU);
	
  return 0;
}

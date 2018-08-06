Dense Prime Cluster GPU (CUDA) Miner for Nexus Prime Channel.

This is a fork of Viz' PrimeSoloMiner project, adding GPU (CUDA) acceleration for sieving.
CUDA additions by cbuchner1, ChrisH

This is also a fork of the cbuchner1 fork, add GPU (CUDA) accelation for fast primality testing.
Additional CUDA additions by BlackJack, Jack M.

****Building Modded Wallet****

In order for 1024-bit testing to work, 1023-bit or less block hashes need to be
generated for the test to fit efficiently within 128 bytes for each prime. This can
be achieved with a modded wallet. It is possible to code in 1024-bit support, but
not desireable, as it will lead to approximately a ~7% drop in speed.

-Recommended wallet version 2.4.6 or better
-FOLLOW COPY INSTRUCTIONS BELOW AND FOLLOW WALLET README FOR REGULAR BUILD PROCESS

->copy/replace uint1024.h in src/hash/
->copy/replace miningserver in src/LLP/

config.ini -change GPU settings regarding seiving, testing

   -nPrimeLimitB:        how many sieving primes
   -nBitArray_Size:      how large the sieving array
           
   -nSieveIterationsLog2 how many bit arrays should be seived before 
                         testing (power of 2) ex 10 = 2^10 = 1024
           
   -nTestLevels          how many chains deep GPU test should go before
                         passing workload to CPU (recommended to not test too deep,
                         or CPU won't be saturated with enough work)
                  
            
offsets.ini -change sieving offsets and sieving primes. probably don't want to
             change this unless you understand what is happening.
             
****COMMAND LINE ARGUMENTS****

   PROGRAM       IP        PORT    GPUs     TIMEOUT
  ./gpuminer 192.168.0.100 9325 0,1,2,3,4,5 15

****TODO****
  -Windows Support
  -Pool Support
  -Mixed GPU (CUDA) Support

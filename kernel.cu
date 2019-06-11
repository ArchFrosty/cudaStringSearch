#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <iostream>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <Windows.h>


using namespace std;

const unsigned PRIME_BASE = 251;

//max size 4 byte prime
const long unsigned PRIME_MOD = 4294967291;
// max substring length 8555711

unsigned const BLOCKS = 40;
unsigned const THREADS = 256;
//maximum size of chunk of data to process at a time
unsigned long long const CHUNK_SIZE = 1000000000;//1GB

long long rabinKarpCPU(char* string, char* substring, unsigned long long fileSize, bool pos);
unsigned int rollingHash(char* string);
__global__ void rabinKarpKernel(unsigned long long stringLenght, char* string, unsigned int substringLenght, char* substring, unsigned int substringHash, unsigned int power, unsigned int* matchesCount, bool pos);

int main(int argc, char** argv)
{
	if (argc == 1 || strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "help") == 0) {
		printf("Substring search in file help\n");
		printf("First argument is -g for GPU calculation or -c for CPU calculation\n");
		printf("Second argument is path to the desired file to be searched\n");
		printf("Third argument is the substring to be searched for\n");
		printf("Fourth argument is either -count for count of matches or -pos for individual positions of the matches; -pos does not work properly fo GPU calculation\n");
		printf("IF using the GPU calculation, the input file has to be at least 15000 times larger that the length of the substring\n");
		return -1;
	}
	if (argc < 5) {
		printf("Not enough arguments\n");
		return -1;
	}
	if (argc > 5) {
		printf("Too many arguments\n");
		return -1;
	}

	//opening and mapping a file to memory

	//open existing file in read mode and get its handle
	HANDLE hFile = CreateFile(argv[2], GENERIC_READ, 0, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	if (hFile == NULL)
	{
		printf("hFile is NULL: last error: %d\n"), GetLastError();
		return (2);
	}

	//get the file size
	LARGE_INTEGER fs;
	GetFileSizeEx(hFile, &fs);
	unsigned long long fileSize = fs.QuadPart;

	//map entire file to to memory in readonly mode 
	HANDLE hMapFile = CreateFileMapping(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
	if (hMapFile == NULL)
	{
		printf("hMapFile is NULL: last error: %d\n"), GetLastError();
		return (2);
	}

	//create a view of the entire memory mapped file and get a pointer to the start adresses
	LPVOID lpMapAddress = MapViewOfFile(hMapFile, FILE_MAP_READ, 0, 0, 0);
	char* data = (char *)lpMapAddress;

	if (lpMapAddress == NULL)
	{
		printf("lpMapAddress is NULL: last error: %d\n"), GetLastError();
		return (2);
	}

	//start timing
	clock_t start;
	double duration;
	start = clock();

	char* stringToFind = argv[3];
	int stringToFindLen = strlen(stringToFind);
	long totMatches = 0;

	bool pos;
	if (strcmp(argv[4], "-pos") == 0) {
		pos = 1;
	}
	else {
		pos = 0;
	}

	//cpu calculation
	if (strcmp(argv[1], "-c") == 0) {
		totMatches = rabinKarpCPU(data, stringToFind, fileSize, pos);
	}
	else if (strcmp(argv[1], "-g") == 0) {
	//GPU calculation
		unsigned int matchesCount = 0;
		unsigned int* d_matchesCount;
		char* d_substring;
		char* d_string;
		unsigned long long alreadyDone = 0;
		unsigned long long currentChunkSize = 0;
		bool dataLeft = 0;

		//allocate memory on device
		if (cudaSuccess != cudaMalloc(&d_matchesCount, sizeof(unsigned int))) {
			printf("Cuda memory allocation failed\n");
			return -1;
		}
		if (cudaSuccess != cudaMalloc(&d_substring, stringToFindLen)) {
			printf("Cuda memory allocation failed\n");
			return -1;
		}
		if (cudaSuccess != cudaMalloc(&d_string, CHUNK_SIZE)) {
			printf("Cuda memory allocation failed\n");
			return -1;
		}

		//main loop
		//separate the memory mapped file into chunks that cen be processed at one time
		while (alreadyDone < fileSize)
		{

			//decide what is the current chunk size and if there is still data left to process
			if (fileSize <= CHUNK_SIZE) {
				currentChunkSize = fileSize;
				dataLeft = 0;
				//printf("setting chunk size to filesize\n");
			}
			else if (fileSize <= (alreadyDone + (CHUNK_SIZE - (stringToFindLen - 1)))) {
				currentChunkSize = fileSize - alreadyDone;
				dataLeft = 0;
				//printf("setting chunk size to filesize - alredy done\n");
			}
			else {
				currentChunkSize = CHUNK_SIZE;
				dataLeft = 1;
				//printf("setting chunk size to CHUNK_SIZE, alredy done:%llu\n", alreadyDone);
			}


			//clear the memory and copy data to the device
			cudaMemset(d_matchesCount, 0, sizeof(unsigned int));
			cudaMemset(d_string, 0, currentChunkSize);
			cudaMemcpy(d_substring, stringToFind, stringToFindLen, cudaMemcpyHostToDevice);
			cudaMemcpy(d_string, data, currentChunkSize, cudaMemcpyHostToDevice);

			//if there is data left move pointer to start of next chunk and increment the done counter, otherwise only increment the done counter
			if (dataLeft) {
				data += currentChunkSize - (stringToFindLen - 1);
				alreadyDone += currentChunkSize - (stringToFindLen - 1);
			}
			else {
				alreadyDone += currentChunkSize;
			}

			unsigned int power = 1;
			for (int i = 0; i < stringToFindLen; i++)
				power = (power * PRIME_BASE) % PRIME_MOD;

			//launch the kernel
			rabinKarpKernel << <BLOCKS, THREADS >> > (currentChunkSize, d_string, stringToFindLen, d_substring, rollingHash(stringToFind), power, d_matchesCount, pos);
			cudaDeviceSynchronize(); // technically not required 
			//copy data back to host
			cudaMemcpy(&matchesCount, d_matchesCount, sizeof(unsigned int), cudaMemcpyDeviceToHost);
			totMatches += matchesCount;
		}
		cudaFree(d_string);
		cudaFree(d_matchesCount);
		cudaFree(d_substring);
	}
	else {
		printf("First argument invalid, use -c for CPU search or -g for GPU search\n");
		return -1;
	}

	duration = (clock() - start) / (double)CLOCKS_PER_SEC;

	if (strcmp(argv[4], "-count") == 0) {
		printf("%d\n", totMatches);
	}
	cout << "Program runtime: " << duration << '\n';


	bool bFlag;
	bFlag = UnmapViewOfFile(lpMapAddress);
	bFlag = CloseHandle(hMapFile); // close the file mapping object

	if (!bFlag)
	{
		printf("\nError %ld occurred closing the mapping object!"), GetLastError();
	}

	bFlag = CloseHandle(hFile);   // close the file itself

	if (!bFlag)
	{
		printf("\nError %ld occurred closing the file!"), GetLastError();
	}

	return 0;
}

unsigned int rollingHash(char* string)
{
	unsigned int ret = 0;
	for (int i = 0; i < strlen(string); i++)
	{
		ret = ret*PRIME_BASE + string[i];
		ret %= PRIME_MOD; //don't overflow
	}
	return ret;
}


__global__ void rabinKarpKernel(unsigned long long stringLenght, char* string, unsigned int substringLenght, char* substring, unsigned int substringHash, unsigned int power, unsigned int* matchesCount, bool pos) {

	int id = blockIdx.x * blockDim.x + threadIdx.x; // id of a thread
	unsigned long chunkSize = (stringLenght + (BLOCKS*THREADS) - 1) / (BLOCKS*THREADS); // amount of data to process by a single thread
	unsigned int rollingHash = 0;
	int mm = 0;

	if (id < stringLenght) {
		for (unsigned long long i = chunkSize*id; i < chunkSize*(id + 1) + substringLenght - 1; i++)
		{
			if (i < stringLenght) {
				//add the last letter
				rollingHash = rollingHash*PRIME_BASE + string[i];
				rollingHash %= PRIME_MOD;

				//remove the first character, if needed
				if (i >= chunkSize*id + substringLenght)
				{
					rollingHash -= power * string[i - substringLenght] % PRIME_MOD;
				}

				//match?
				if (i >= substringLenght - 1 && rollingHash == substringHash) {
					bool match = 1;
					for (int j = 0; j < substringLenght; j++)
					{
						if (string[i - j] != substring[substringLenght - j - 1]) {
							match = 0;
						}
					}
					if (match) {
						mm = atomicAdd(matchesCount, 1); // atomicly incerement a found string counter
						if (pos) {
							printf("Match %d found by thread %d at: %llu\n",mm ,id , i - substringLenght);// WHY DO YOU HATE ME
						}
					}
				}
			}
		}
	}
}

long long rabinKarpCPU(char* string, char* substring, unsigned long long fileSize, bool positions)
{
	unsigned int substringHash = rollingHash(substring);
	unsigned int rollingHash = 0;
	long long count = 0;

	unsigned long long stringLength = fileSize;
	unsigned long long substringLength = strlen(substring);

	//this is the max power for hash, meaning we can access the topmost hashed character just by multiplying with this power
	unsigned int power = 1;
	for (int i = 0; i < substringLength; i++)
		power = (power * PRIME_BASE) % PRIME_MOD;

	for (unsigned long long i = 0; i < stringLength; i++)
	{
		//add the last letter
		rollingHash = rollingHash*PRIME_BASE + string[i];
		rollingHash %= PRIME_MOD;

		//remove the first character, if needed
		if (i >= substringLength)
		{
			rollingHash -= power * string[i - substringLength] % PRIME_MOD;
		}

		//match?
		if (i >= substringLength - 1 && rollingHash == substringHash) {

			bool match = 1;
			for (int j = 0; j < substringLength; j++)
			{
				if (string[i - j] != substring[substringLength - j - 1]) {
					match = 0;
				}
			}
			if (match) {
				if (positions) {
					printf("Match found at postition: %llu\n", i - substringLength);
				}
				count++;
			}
		}
	}
	return count;
}



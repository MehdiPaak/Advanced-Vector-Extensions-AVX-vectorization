/*-------------------------------------------------------
avx_test.cpp
 Driver program for AVX test
 Tests:
	- memory alignment
	- subtract operations
	- Distance calculation
	- timing of distance calculation for a matrix

TODO:
	Test driver is ad-hoc, modify for better testing
By: Mehdi Paak
---------------------------------------------------------*/

#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include "AVXFunc.h"

using namespace std;
int main()
{
	// Size of vectors
	const int nN = 67;
	// Number of vectors for matrix test
	const int nM = 10000;

	float *afV1, *afV2, *afV3, *afV3_R;
	// Alignment 32 bit (since using floats)
	const size_t  nAlignment = 32;

	// Using _aligned_malloc
	afV1 = (float *) _aligned_malloc(nN * sizeof(float), nAlignment);
	afV2 = (float *) _aligned_malloc(nN * sizeof(float), nAlignment);
	afV3 = (float*)_aligned_malloc(nN * sizeof(float), nAlignment);
	afV3_R = (float *) _aligned_malloc(nN * sizeof(float), nAlignment);

	if (afV1 == NULL || afV2 == NULL)
	{
		printf_s("Error allocation aligned memory.");
		return -1;
	}

	printf_s("----------\n");
	printf_s("Vector dimension = %d\n", nN);
	printf_s("number of vectors = %d\n", nM);
	printf_s("----------\n");
	// Test memory alignment
	if (IsMemAligned((void*)afV1, nAlignment))
		printf_s("Memory pointer, %p, is aligned on %zu\n", afV1, nAlignment);
	else
		printf_s("Memory pointer, %p, is not aligned on %zu\n", afV1, nAlignment);

	// fill the array using rendom numbers
	srand(117);
	for (int i = 0; i < nN; ++i)
	{
		afV1[i] = (float) (rand() % 10);
		afV2[i] = (float) (rand() % 10);
	}

	// Testing vector subtraction
	// standard
	SubVecs(nN, afV1, afV2, afV3);
	// AVX
	SubVecsAvx(nN, afV1, afV2, afV3_R);
	float SqErr = CompareRes(nN, afV3, afV3_R);
	// Error must be zero
	printf_s("----------\n");
	printf_s("AVX-Standard Subtraction squared error = %f\n", SqErr);

	// Testing Distance
	// standard
	float fDistReg = DistReg(nN, afV1, afV2);
	// AVX
	float fDistAvx = DistAvx(nN, afV1, afV2);
	printf_s("\n--Distance Calculation--\n");
	printf_s(" DistReg = %f\n DistAvx = %f\n", fDistReg, fDistAvx);

	// Free mem
	_aligned_free(afV1);
	_aligned_free(afV2);
	_aligned_free(afV3);
	_aligned_free(afV3_R);

	// Tesing Euclidean distance for matrix
	double T1 = TestEucliDistAvx(nM, nN,DistReg);
	double T2 = TestEucliDistAvx(nM, nN,DistAvx);
	// print timing
	printf_s(" Elapsed time standard = %f sec\n Elapsed time AVX      = %f sec\n", T1, T2);

	return 0;
}


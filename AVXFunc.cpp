/*-------------------------------------------------------
AVXFunc.cpp 
Implementation of Functions used for testing vectorized operations
using AVX instruction intrinsics.

Note:
Vectorization and Register level programming is fun!

By: Mehdi Paak
---------------------------------------------------------*/

#include <math.h>   
#include <cstdlib>
#include <iostream>

//--intrinsic functions --
#include <immintrin.h>

//-- for high_resolution_clock --
#include <chrono>         

//-- utility functions --
#include "AVXFunc.h"

#define MEM_ALIGNED

//================================================================
// DESCRIPTION:
//   Check if the allocated memory is aligned or not.
//   This is important for vectorized operations. 
// IN:
//   ptr: void, pointer to memory
//   nAlignment: int, desired memory alignment e.g. 32, 64 etc
//
// RETURN:
//  bool
//================================================================
bool IsMemAligned(void* ptr, size_t nAlignment)
{
	if ((unsigned long long) ptr % nAlignment == 0)
		return true;
	else
	    return false;
}

//================================================================
// DESCRIPTION:
//   Subtract operation: V3=V1-V2
//
// INPUT:
//	nN: int, size of vectors
//  afV1: float, array for V1
//  afV2: float, array for V2
//
// OUTPUT:
//  afV3: float, array for V3
//
// RETURN:
//
// NOTE:
//
//================================================================
void SubVecs(const int nN, const float* afV1, const float* afV2, float* afV3)
{
	for (int i = 0; i < nN; ++i)
	{
		afV3[i] = afV1[i] - afV2[i];
	}

}

//================================================================
// DESCRIPTION:
//   AVX Subtract operation: V3=V1-V2
//
// INPUT:
//	nN: int, size of vectors
//  afV1: float, array for V1
//  afV2: float, array for V2
//
// OUTPUT:
//  afV3: float, array for V3
//
// RETURN:
//
// NOTE:
//  8 floats (32 bits) fit into 256 bits register 
//================================================================
void SubVecsAvx(const int nN, const float* afV1, const float* afV2, float* afV3)
{

	for (size_t j = 0; j < nN / 8; ++j)
	{
		__m256 R_V1 = _mm256_load_ps(afV1 + 8 * j); // load
		__m256 R_V2 = _mm256_load_ps(afV2 + 8 * j); // load
		R_V2 = _mm256_sub_ps(R_V1, R_V2);           // subtract
		_mm256_store_ps(afV3 + 8 * j, R_V2);        // store
	}

	// Compute the rest
	for (int j = nN - nN % 8; j < nN; ++j)
	{
		afV3[j] = afV1[j] - afV2[j];
	}

}

//================================================================
// DESCRIPTION:
//   computes sum of squared error (SSE) between V1 and V2
//   
// INPUT:
//	nN: size of vectors
//  afV1: array of float
//  afV2: array of float
// OUTPUT:
// 
// RETURN:
//   SSE
// NOTE:
//
//================================================================
float CompareRes(const int nN, const float* afV1, const float* afV2)
{
	float SqErr = 0.0f;
	
	for (int i = 0; i < nN; ++i)
	{
		const float fTmp = afV1[i] - afV2[i];
		SqErr += fTmp * fTmp;
	}

	return SqErr;
}

//================================================================
// DESCRIPTION:
//   Compute Euclidean distance between vectors V1 and V2,
//   using standard definition.
// INPUT:
//   afV1: array of float
//   afV2: array of float
// OUTPUT:
// 
// RETURN:
//   Distance
// NOTE:
//
//================================================================
float DistReg(const int nN, const float* afV1, const float* afV2)
{
	float dSum = 0.0f;
	for (int i = 0; i < nN; ++i)
	{
		const float fTmp = afV1[i] - afV2[i];
		dSum += fTmp * fTmp;
	}
	return sqrtf(dSum);
}

//================================================================
// DESCRIPTION:
//		Reduce sum operation on 256 bit register,
//      adds 8 floats in __m256 bits.
// INPUT:
//   afV: __256, 8 floats in vector afV
//
// OUTPUT:
// 
// RETURN:
//   Reduce sum
//
// NOTE:
//
//================================================================
float ReduceAvx(__m256 afV)
{
	afV = _mm256_hadd_ps(afV, afV);
	afV = _mm256_hadd_ps(afV, afV);

	return ((float*)& afV)[0] + ((float*)& afV)[4];
}

//================================================================
// DESCRIPTION:
//   Compute Euclidean distance between vectors V1 and V2,
//   using AVX instructions.
// INPUT:
//   afV1: array of float
//   afV2: array of float
// OUTPUT:
// 
// RETURN:
//   Distance
// NOTE:
//
//================================================================
float DistAvx(const int nN, const float* afV1, const float* afV2)
{
	__m256 R_Sum = _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
	for (size_t j = 0; j < nN / 8; ++j)
	{
		__m256 R_V1 = _mm256_load_ps(afV1 + 8 * j);
		__m256 R_V2 = _mm256_load_ps(afV2 + 8 * j);
		R_V2 = _mm256_sub_ps(R_V1, R_V2);
		R_V2 = _mm256_mul_ps(R_V2, R_V2);
		R_Sum = _mm256_add_ps(R_Sum, R_V2);
	}

	// compute remainder
	float dSum = 0.0f;
	for (int i = nN - nN % 8; i < nN; ++i)
	{
		const float fTmp = afV1[i] - afV2[i];
		dSum += fTmp * fTmp;
	}

	// Reduce sum values in the register and sum of remainders
	return sqrtf(ReduceAvx(R_Sum) + dSum);
}
//================================================================
// DESCRIPTION:
//  Compute Euclidean distance between rows of a matrix MxN.
//  For M rows there will be M(M-1)/2 distances.
//  dist = sqrt(sum((XI-XJ).^2,2));            
// IN:
//	afVecMat: array of floats, Matrix MxN (M vectors of size N)
//	nNumVecs: number of vectors, M
//	nSizeVec: size of vectors N
//  func: pointer to distance function
//
// OUT:
//	afDists: array of float, distances. M(M-1)/2 
//
// RETURN:
//
// NOTE:
//  
//================================================================
void CalcEucdistMat(float* afVecMat,
					const size_t nNumVecs,
					const size_t nSizeVec,
					float* afDists,
					float(*func)(const int, const float*, const float*))
{
	if (!afVecMat)
	{
		return;
	}

	size_t k = 0;
	for (size_t i = 0; i < nNumVecs; ++i)
	{

		for (size_t j = i + 1; j < nNumVecs; ++j)
		{

			const float dSum = func(nSizeVec, afVecMat + i * nSizeVec, afVecMat + j * nSizeVec);

			afDists[k++] = dSum;
		}
	}
}

//================================================================
// DESCRIPTION:
//  Test function for CalcEucdistMat. A random matrix MxN matrix
//  is generated and distances are calculated using the input function.
//  Matrix		   [--V1--;
//					--V2--;
//					-....-,
//					--Vm--]
// IN:
//	nNumVecs: number of vectors, M
//	nSizeVec: size of vectors N
//  func: pointer to distance function
//
// OUT:
//
// RETURN:
//   elapsed time in seconds
// NOTE:
//  
//================================================================
double TestEucliDistAvx(const size_t nNumVecs, 
						const size_t nSizeVec, 
						float (*func)(const int, const float*, const float*))
{

    //Random seed
	srand(113);
	// Matrix Dims
	const size_t M = nNumVecs, N = nSizeVec;

	// Allocate Mem
#ifdef MEM_ALIGNED
	float* afVMat = (float*)_aligned_malloc(M * N * sizeof(float), 32);
	float* afDist = (float*)_aligned_malloc((M*(M - 1) / 2) * sizeof(*afDist), 32);
#else
	float* afVMat = (float*) malloc(M * N * sizeof(*afVMat));
	float* afDist = (float*) malloc((M*(M-1)/2) * sizeof(*afDist));
#endif

	if (!afVMat || !afDist)
		return -1.0;

	// Initialize with random numbers between 0 and 10
	for (int i = 0; i < N * M; ++i)
	{
		const float val = (float) (rand() % 10);
		afVMat[i] = val;
	}

	//// Print matrix
	//printf_s("\n");
	//for (int i = 0; i < M; ++i)
	//{
	//	for (int j = 0; j < N; ++j)
	//	{
	//		printf_s("%f ", afVMat[i * n + j]);
	//	}
	//	printf_s("\n");
	//}

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();
	// Calculate distances
	CalcEucdistMat(afVMat, M, N, afDist, func);
	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;


	//	printf_s("\n");
	//	for (int i = 0; i < 3; ++i)
	//		printf_s("%f ", d[i]);

#ifdef MEM_ALIGNED
	_aligned_free(afVMat);
	_aligned_free(afDist);
#else
	free(afVMat);
	free(afDist);
#endif

	return elapsed.count(); //secs
}


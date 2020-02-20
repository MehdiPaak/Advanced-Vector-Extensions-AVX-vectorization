/*-------------------------------------------------------
AVXFunc.h
Functions used for testing vectorized operations 
using AVX instruction intrinsics. 

Note:
	Vectorization and Register level programming is fun!

By: Mehdi Paak
---------------------------------------------------------*/

#ifndef UTILITY_H
#define UTILITY_H

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
bool IsMemAligned(void* ptr, size_t nAlignment);

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
void SubVecs(const int nN, 
			 const float* afV1, 
			 const float* afV2, 
			 float* afV3);

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
//
//================================================================
void SubVecsAvx(const int nN, 
				const float* afV1,
				const float* afV2, 
				float* afV3);

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
float ReduceAvx(__m256 afV);

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
float CompareRes(const int nN, 
				 const float* afV1, 
				 const float* afV2);

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
float DistAvx(const int nN, 
			  const float* afV1, 
			  const float* afV2);

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
float DistReg(const int nN, 
			  const float* afV1, 
			  const float* afV2);

//================================================================
// DESCRIPTION:
//  Compute Euclidean distance between rows of a matrix MxN.
//  For M rows there will be M(M-1)/2 distances.
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
					float(*func)(const int, const float*, const float*));

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
	                    float (*func)(const int, const float*, const float*));

//================================================================
#endif // UTILITY_H
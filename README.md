## Vectorization
Even if you don't have access to CPU  and/or GPU clusters, you could still benefit from
vectorization to achieve palatalization and performance improvement. This can be done 
thanks to existence of wide registers on  CPUs. Modern CPUs have multiple registers with 256 or 512 bits
in width. 
Many scientific computations require one operation to be performed on multiple data; say addition or subtraction 
of two vectors, dot product etc. In these situations, data from vectors can be taken in to registers in chunks
and one instruction is applied to them--single instruction multiple data (SIMD).

> Modern compilers with advanced optimizations usually take care of these types of SIMD, behind the scene.
> Math library functions have various overloads for various architectures, optimized for vectorization.

There are situation where you might be interested to write vectorized codes and manipulate registers yourself.
To do this, we have to code in **Assembly!** or machine language which is no mean feat. Here,
intrinsic functions come to rescue. They can be used  directly in the source code  in lieu of assembly.

This program contains AVX functionalities to calculate the Euclidean distance between vectors. 

##  Advanced Vector Extensions (AVX instructions)
Intrinsic functions for YMM registers can be found in [Intel's intrinsics guide](https://software.intel.com/sites/landingpage/IntrinsicsGuide/#techs=AVX).

##  How to use
     Build the solution and run 'avx_test.exe'.
##  Author
Mehdi Paak

##  License

This project is licensed under the MIT License - see the  [LICENSE]()  file for details. For pedagogical purposes, please, do not submit this code as a solution for course assignments,exams and interview case studies.
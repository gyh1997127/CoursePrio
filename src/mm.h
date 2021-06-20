#ifndef _H_GEMM_H_
#define _H_GEMM_H_

#define NI 2048
#define NJ 2048
#define NK 2048
//Standard Libraries
#include <stdio.h>
#include <stdlib.h>

//Define compute data type
#define TYPE float
//#define unroll_size 128


#define MatrixSize NI
//Specify row/column sizes
#define row_size MatrixSize
#define col_size MatrixSize

//specify tile_size
#define tile_size 128

void kernel_gemm (float C[NI*NJ], float A[NI*NK], float B[NK*NJ], const float alpha, const float beta);

#endif // _H_GEMM_H_


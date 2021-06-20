
#define NI 1024
#define NJ 1024
#define NK 1024
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
#define tile_size 256
#define MAX_SIZE tile_size

void kernel_gemm (float C[NI*NJ], float A[NI*NK], float B[NK*NJ], const float alpha, const float beta);



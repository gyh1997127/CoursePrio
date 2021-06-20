
#define NI 64
#define NJ 64
#define NK 64
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
#define tile_size 32
#define MAX_SIZE tile_size

void kernel_gemm (float C[NI*NJ], float A[NI*NK], float B[NK*NJ], const float alpha, const float beta);

void read_A(float A[NI*NK], float local_A[tile_size*tile_size], int i, int k, int alpha);

void read_B(float B[NK*NJ], float local_B[tile_size*tile_size], int k, int j);

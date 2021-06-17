#include <algorithm>
#include <stdio.h>
#include <stdlib.h>

#define NI 2048
#define NJ 2048
#define NK 2048
#define BSize 32

void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta);
void kernel_gemm_serial_opt (float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta); 
void kernel_gemm_tiled (float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta);


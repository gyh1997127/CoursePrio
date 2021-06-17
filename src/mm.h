#include <stdio.h>
#include <stdlib.h>

#define NI 5
#define NJ 5
#define NK 5
#define BSize 1
#define FACTOR NI/BSize

void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta);


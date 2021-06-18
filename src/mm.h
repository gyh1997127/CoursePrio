//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>

#define NI 2048
#define NJ 2048
#define NK 2048
#define BSize 32
#define FACTOR NI/BSize

void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta);
float print_array_sum(float C[NI*NJ]);
void print_array(float C[NI*NJ]);

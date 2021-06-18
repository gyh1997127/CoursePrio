//#include <stdio.h>
//#include <stdlib.h>
#include <iostream>

#define NI 5
#define NJ 5
#define NK 5
#define BSize 1
#define FACTOR NI/BSize

void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta);
float print_array_sum(float C[NI*NJ]);
void print_array(float C[NI*NJ]);

#include "mm.h"
//#include <stdio.h>

void print_array(float C[NI * NJ]) {
  int i, j;

  for (i = 0; i < NI; i++)
    //for (j = 0; j < NJ; j++)
      //printf("C[%d][%d] = %f\n", i, j, C[i * NJ + j]);
      printf("%f, %f, %f\n", C[i*NJ+0], C[i*NJ+1], C[i*NJ+2]);
  printf("\n");
}

float print_array_sum(float C[NI * NJ]) {
  int i, j;

  float sum = 0.0;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i * NJ + j];

  printf("sum of C array = %f\n", sum);
  return sum;
}

void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ],
                 float alpha, float beta) {
#pragma HLS ARRAY_PARTITION variable = C dim = 1 block  factor = 64
#pragma HLS ARRAY_PARTITION variable = A dim = 1 block factor = 64
#pragma HLS ARRAY_PARTITION variable = B dim = 1 cyclic factor = 64
#pragma HLS dataflow
#ifndef __SYNTHESIS__
  std::cout << "kernel_gemm\n";
#endif
  int i, j, k;
  float temp_sum[NJ] = {0};
  print_array(A);
  print_array(B);

   //=> Form C := alpha*A*B + beta*C,
   //A is NIxNK
   //B is NKxNJ
   //C is NIxNJ
  InitC_i:for (i = 0; i < NI; i++) {
    InitC_j:for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= beta;
    }
  }
  print_array(C);
  MM_i:for (i = 0; i < NI; i++) {
   MM_k:for (k = 0; k < NK; k++) {
     #pragma HLS unroll factor=64
     MM_j:for (j = 0; j < NJ; j++) {
       #pragma HLS PIPELINE II=1
       float result = (k==0) ? 0 : temp_sum[j];
       result += alpha * A[i*NK+k] * B[k*NJ+j];
       temp_sum[j] = result;
       if (k == NK - 1) 
	 C[i*NJ+j] += temp_sum[j];
       //C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
     }
   }
  }

}

#include "mm.h"

void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ],
                 float alpha, float beta) {
  int i, j, k;

  // => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ

  InitC_i:for (i = 0; i < NI; i++) {
    InitC_j:for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= beta;
    }

  MM_j:for (j = 0; j < NJ; j++) {
    MM_k:for (k = 0; k < NK; ++k) {
        C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}



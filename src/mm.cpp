#include "mm.h"

/* Sequential computational kernel. The whole function is timed,
   including the call and return. */
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

void kernel_gemm_serial_opt(float C[NI * NJ], float A[NI * NK],
                            float B[NK * NJ], float alpha, float beta) {
  int i, j, k;

  // => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ

  // Initialising C matrix: C=C*Beta
  InitC_i:for (i = 0; i < NI; i++) {
    InitC_j:for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= beta;
    }
  }

  // Computing matrix product: C=Alpha*A*B + C
  MM_i:for (i = 0; i < NI; i++) {
    MM_k:for (k = 0; k < NK; ++k) { // re-ordered the loop here to I-K-J for better locality on A,B and C
      MM_j:for (j = 0; j < NJ; j++) {
          C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}

/* Sequential loop-tiled kernel_gemm. The whole function is timed,
   including the call and return. */
void kernel_gemm_tiled(float C[NI * NJ], float A[NI * NK], float B[NK * NJ],
                       float alpha, float beta) {
  int i, j, k;
  int ii, jj, kk;
  int nthreads = 6;

  // => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ

  // Initialisation of matrix C: C=C*Beta
  InitC_i:for (int i = 0; i < NK; i += BSize) {
    InitC_j:for (int j = 0; j < NJ; j += BSize) {
      InitC_ii:for (ii = 0; ii < BSize; ii++) { // Applying loop tiling to increase cache hits
        InitC_jj:for (jj = 0; jj < BSize; jj++) {
            C[(i + ii) * NJ + (j + jj)] *= beta;
        }
      }
    }
  }

  // Main matrix multiplication kernel: C = C + A*B*Alpha
  MM_i:for (i = 0; i < NI; i += BSize) {
    MM_k:for (k = 0; k < NK; k += BSize) { // changing loop order to I->K->J from I->J->K to exploit
      MM_j:for (j = 0; j < NJ; j += BSize) {
        MM_ii:for (ii = i; ii < i + BSize; ii++) { // Applying loop tiling to increase the cache hits
          MM_kk:for (kk = k; kk < k + BSize; kk++) { // Loop ordering for inner-most loops is kept as
            MM_jj:for (jj = j; jj < j + BSize; jj++) {
                C[ii * NJ + jj] += alpha * A[ii * NK + kk] * B[kk * NJ + jj];
            }
          }
        }
      }
    }
  }
}

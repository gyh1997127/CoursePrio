#include "mm_test.h"

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

void init_array(float C[NI * NJ], float A[NI * NK], float B[NK * NJ]) {
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      C[i * NJ + j] = (float)((i * j + 1) % NI) / NI;
  for (i = 0; i < NI; i++)
    for (j = 0; j < NK; j++)
      A[i * NK + j] = (float)(i * (j + 1) % NK) / NK;
  for (i = 0; i < NK; i++)
    for (j = 0; j < NJ; j++)
      B[i * NJ + j] = (float)(i * (j + 2) % NJ) / NJ;
}

static void kernel_gemm_tiled(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta) {
  int i, j, k;
  int ii, jj, kk;
  for (i = 0; i < NI; i+=BSize) {
    for (j = 0; j < NJ; j+=BSize) {
      for (ii = i; ii < i+BSize; ii++) { // Applying loop tiling to increase cache hits
        for (jj = j; jj < j+BSize; jj++) {
          C[ii * NJ + jj] *= beta;
        }
      }
    }
  }

  for (i = 0; i < NI; i += BSize) {
    for (k = 0; k < NK; k += BSize) { // changing loop order to I->K->J from I->J->K to exploit
      for (j = 0; j < NJ; j += BSize) {
        for (ii = i; ii < i + BSize; ii++) { // Applying loop tiling to increase the cache hits
          for (kk = k; kk < k + BSize; kk++) { // Loop ordering for inner-most loops is kept as
            for (jj = j; jj < j + BSize; jj++) {
              C[ii * NJ + jj] += alpha * A[ii * NK + kk] * B[kk * NJ + jj];
            }
          }
        }
      }
    }
  }
}

static void kernel_gemm_serial_opt(float C[NI * NJ], float A[NI * NK], float B[NK * NJ], float alpha, float beta) {
  int i, j, k;
  std::cout << "serial opted\n";
  //print_array(A);
  //print_array(B);
  // => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ
  InitC_i:for (i = 0; i < NI; i++) {
    InitC_j:for (j = 0; j < NJ; j++) {
      C[i * NJ + j] *= beta;
    }
  }
  MM_i:for (i = 0; i < NI; i++) {
    MM_k:for (k = 0; k < NK; ++k) { // re-ordered the loop here to I-K-J for better locality on A,B and C
      MM_j:for (j = 0; j < NJ; j++) {
          C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
      }
    }
  }
}

int main(int argc, char **argv) {
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI * NK * sizeof(float));
  float *B = (float *)malloc(NK * NJ * sizeof(float));
  float *C = (float *)malloc(NI * NJ * sizeof(float));

  float *A_ref = (float *)malloc(NI * NK * sizeof(float));
  float *B_ref = (float *)malloc(NK * NJ * sizeof(float));
  float *C_ref = (float *)malloc(NI * NJ * sizeof(float));
  /* Initialize array(s). */
  init_array(C, A, B);
  init_array(C_ref, A_ref, B_ref);

  /* Run kernel. */
  kernel_gemm(C,A,B,1.5,2.5);
  //print_array(C);
  float kernel_sum = print_array_sum(C);

  kernel_gemm_serial_opt(C_ref,A_ref,B_ref,1.5,2.5);
  //kernel_gemm_tiled(C_ref, A_ref, B_ref, 1.5, 2.5);
  //print_array(C_ref);
  float ref_sum = print_array_sum(C_ref);

  if (kernel_sum == ref_sum) {
    std::cout << "passed\n";
    free(A);
    free(B);
    free(C);
    return 0;
  } else {
    printf("Result Error!\n");
    free(A);
    free(B);
    free(C);
    return 1;
  }
  return 0;
}

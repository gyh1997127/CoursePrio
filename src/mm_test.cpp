#include "mm.h"

/* Array initialization. */
static void init_array(float C[NI * NJ], float A[NI * NK], float B[NK * NJ]) {
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

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static void print_array(float C[NI * NJ]) {
  int i, j;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      printf("C[%d][%d] = %f\n", i, j, C[i * NJ + j]);
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
float print_array_sum(float C[NI * NJ]) {
  int i, j;

  float sum = 0.0;

  for (i = 0; i < NI; i++)
    for (j = 0; j < NJ; j++)
      sum += C[i * NJ + j];

  printf("sum of C array = %f\n", sum);
  return sum;
}


int main(int argc, char **argv) {
  /* Variable declaration/allocation. */
  float *A = (float *)malloc(NI * NK * sizeof(float));
  float *B = (float *)malloc(NK * NJ * sizeof(float));
  float *C = (float *)malloc(NI * NJ * sizeof(float));

  /* Initialize array(s). */
  init_array(C, A, B);

  /* Run kernel. */
  kernel_gemm(C,A,B,1.5,2.5);
  //kernel_gemm_serial_opt(C,A,B,1.5,2.5);
  //kernel_gemm_tiled(C, A, B, 1.5, 2.5);

  /* Print results. */
  //print_array(C);
  //print_array_sum(C);
  if (print_array_sum(C) == 3212133376.000000) {
    printf("Passed\n");
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
}

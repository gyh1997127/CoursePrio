#include "mm.h"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

// 0.34s
//void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ],
                 //float alpha, float beta) {
  //float temp_sum[NJ] = {0};
  //#pragma HLS ARRAY_PARTITION variable = C dim = 1 block factor = 64
  //#pragma HLS ARRAY_PARTITION variable = A dim = 1 block factor = 64
  //#pragma HLS ARRAY_PARTITION variable = B dim = 1 cyclic factor = 64
  //#pragma HLS ARRAY_PARTITION variable = temp_sum dim = 1 block factor 64
  //#pragma HLS dataflow

//#ifndef __SYNTHESIS__
  //std::cout << "kernel_gemm\n";
//#endif
  //int i, j, k;

  ////=> Form C := alpha*A*B + beta*C,
  //// A is NIxNK
  //// B is NKxNJ
  //// C is NIxNJ
//InitC_i:
  //for (i = 0; i < NI; i++) {
  //#pragma HLS pipeline II = 1
  //InitC_j:
    //for (j = 0; j < NJ; j++) {
  //#pragma HLS loop_flatten
      //C[i * NJ + j] *= beta;
    //}
  //}
//MM_i:
  //for (i = 0; i < NI; i++) {
  //MM_k:
    //for (k = 0; k < NK; k++) {
    //#pragma HLS unroll
    //MM_j:
      //for (j = 0; j < NJ; j++) {
        //#pragma HLS PIPELINE II = 1
        //float result = (k == 0) ? 0 : temp_sum[j];
        //result += alpha * A[i * NK + k] * B[k * NJ + j];
        //temp_sum[j] = result;
        //if (k == NK - 1)
          //C[i * NJ + j] += temp_sum[j];
        //// C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
      //}
    //}
  //}
//}

void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ],
                 float alpha, float beta) {
  //#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem
  //#pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem
  //#pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem
  //#pragma HLS INTERFACE s_axilite port=A bundle=control
  //#pragma HLS INTERFACE s_axilite port=B bundle=control
  //#pragma HLS INTERFACE s_axilite port=C bundle=control
  //#pragma HLS INTERFACE s_axilite port=return bundle=control

  TYPE local_A[tile_size][tile_size];
  TYPE local_B[tile_size][tile_size];
  #pragma HLS ARRAY_PARTITION variable = local_B complete dim = 2
  TYPE local_C[tile_size][tile_size];
  #pragma HLS ARRAY_PARTITION variable = local_C complete dim = 2
  // array partition
  #pragma HLS DEPENDENCE variable = "local_C" inter false

ROW_PARTITION_L:
  for (int i = 0; i < row_size; i += tile_size)
  COL_PARTITION_L:
    for (int j = 0; j < col_size; j += tile_size) {

    LOAD_INIT_TILE_C:
      for (int ii = 0; ii < tile_size; ii++)
        for (int jj = 0; jj < tile_size; jj++)
          #pragma HLS PIPELINE
          local_C[ii][jj] = beta * C[(i + ii) * col_size + (j + jj)];

    LOAD_AB_AND_COMPUTE:
      for (int k = 0; k < col_size; k += tile_size) {
      LOAD_INIT_TILE_A:
        for (int ii = 0; ii < tile_size; ii++)
          for (int kk = 0; kk < tile_size; kk++)
            #pragma HLS PIPELINE
            local_A[ii][kk] = alpha * A[(i + ii) * col_size + (k + kk)];

      LOAD_B:
        for (int kk = 0; kk < tile_size; kk++)
          for (int jj = 0; jj < tile_size; jj++)
            #pragma HLS PIPELINE
            local_B[kk][jj] = B[(k + kk) * col_size + (j + jj)];

      COMPUTE_TILE_LOOP:
        for (int kk = 0; kk < tile_size; kk++)
          for (int ii = 0; ii < tile_size; ii++)
            #pragma HLS PIPELINE II = 1
            for (int jj = 0; jj < tile_size; jj++)
              #pragma HLS UNROLL
              local_C[ii][jj] += local_A[ii][kk] * local_B[kk][jj];
      }

    STORE_TILE_LOOP:
      for (int ii = 0; ii < tile_size; ii++)
        for (int jj = 0; jj < tile_size; jj++)
          #pragma HLS PIPELINE
          C[(i + ii) * col_size + (j + jj)] = local_C[ii][jj];
    }
  return;
}

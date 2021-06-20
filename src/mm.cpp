#include "mm.h"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

//0.564
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
  #pragma HLS ARRAY_PARTITION variable = local_A complete dim = 2
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
    #pragma HLS loop_flatten off

    LOAD_INIT_TILE_C:
      for (int ii = 0; ii < tile_size; ii++)
        for (int jj = 0; jj < tile_size; jj++)
          #pragma HLS PIPELINE
          local_C[ii][jj] = beta * C[(i + ii) * col_size + (j + jj)];

    LOAD_AB_AND_COMPUTE:
      for (int k = 0; k < col_size; k += tile_size) {
        //#pragma HLS loop_flatten off
      //LOAD_INIT_TILE_A:
        //for (int ii = 0; ii < tile_size; ii++)
          //for (int kk = 0; kk < tile_size; kk++)
            //#pragma HLS PIPELINE
            //#pragma HLS loop_flatten 
            //local_A[ii][kk] = alpha * A[(i + ii) * col_size + (k + kk)];

      LOAD:
        for (int kk = 0; kk < tile_size; kk++) {
        #pragma HLS dataflow
        //#pragma HLS PIPELINE
        LOAD_B:
          for (int jj = 0; jj < tile_size; jj++)
            #pragma HLS PIPELINE
            local_B[kk][jj] = B[(k + kk) * col_size + (j + jj)];
        LOAD_A:
          for (int ii = 0; ii < tile_size; ii++)
            #pragma HLS PIPELINE
            local_A[ii][kk] = alpha * A[(i + ii) * col_size + (k + kk)];
        }

      COMPUTE_TILE_LOOP:
        for (int kk = 0; kk < tile_size; kk++)
          for (int ii = 0; ii < tile_size; ii++)
            #pragma HLS PIPELINE II = 1
            for (int jj = 0; jj < tile_size; jj++) {
              #pragma HLS UNROLL
              local_C[ii][jj] += local_A[ii][kk] * local_B[kk][jj];
            }
      }

    STORE_TILE_LOOP:
      for (int ii = 0; ii < tile_size; ii++)
        for (int jj = 0; jj < tile_size; jj++)
          #pragma HLS PIPELINE
          C[(i + ii) * col_size + (j + jj)] = local_C[ii][jj];
    }
  return;
}

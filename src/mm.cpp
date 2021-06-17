#include "mm.h"
void kernel_gemm(float C[NI * NJ], float A[NI * NK], float B[NK * NJ],
                 float alpha, float beta) {
  #pragma HLS ARRAY_PARTITION variable=C dim=1 cyclic factor=64
  #pragma HLS ARRAY_PARTITION variable=A dim=1 cyclic factor=64
  #pragma HLS ARRAY_PARTITION variable=B dim=1 cyclic factor=64
  int i, j, k;
  float temp_sum[NJ];

  // => Form C := alpha*A*B + beta*C,
  // A is NIxNK
  // B is NKxNJ
  // C is NIxNJ
  
  //MM_i:for (i = 0; i < NI; i++) {
  //  MM_k:for (k = 0; k < NK; ++k) {
  //    C[i * NJ + k] *= beta;
  //    MM_j:for (j = 0; j < NJ; j++) {
  //      #pragma HLS PIPELINE II=1
  //      int temp = (k==0) ? 0 : temp_sum[j];
  //      temp += alpha * A[i * NK + k] * B[k * NJ + j];
  //      temp_sum[j] = temp;
  //      if (k == NJ - 1) C[i * NJ + j] = temp;
  //      //C[i * NJ + j] += alpha * A[i * NK + k] * B[k * NJ + j];
  //    }
  //  }
  //}

  K:for(k=0; k<NK; k++) {
      I:for (i=0; i<NI; i++) {
          J: for (j=0; j<NJ; j++) {
	       int last = (k==0) ? C[i * NJ + j]*beta : C[i * NJ + j];
	       //update new sum
	       int a_val = (i<NI && k<NK) ? A[i * NK + k] : 0;
	       int b_val = (k<NK && j<NJ) ? B[k * NJ + j] : 0;
	       int temp = alpha * a_val * b_val;
	       //update C
	       C[i * NJ + j] = temp;
	  }
      } 
     }
}


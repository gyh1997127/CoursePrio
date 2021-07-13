#include "libwb/wb.h"
#include "my_timer.h"
#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define BLUR_SIZE 21
#define X_NTHREADS 512
#define Y_NTHREADS 512

///////////////////////////////////////////////////////
__device__ void blur_x(float *out, float *in, int width, int height) {
  //int r = BLUR_SIZE;
  float scale = 1.0 / (float)((BLUR_SIZE << 1) + 1); // 1/(2r+1)

  /////////////////////
  // col [0, BLUR_SIZE]
  float t;
  t = in[0] * BLUR_SIZE; // padding with in[0]

  // accumulate BLUR_SIZE pixels to in[0]'s right
  // and average out
  for (int x = 0; x < (BLUR_SIZE + 1); x++) {
    t += in[x];
  }
  out[0] = t * scale;

  // take care of 0<x<BLUR_SIZE+1
  for (int x = 1; x < (BLUR_SIZE + 1); x++) {
    t += in[x + BLUR_SIZE];
    t -= in[0];  
    out[x] = t * scale;
  }

  /////////////////////
  // col [BLUR_SIZE+1, width-BLURSIZE]
  for (int x = (BLUR_SIZE + 1); x < width - BLUR_SIZE; x++) {
    t += in[x + BLUR_SIZE];
    t -= in[x - BLUR_SIZE - 1];
    out[x] = t * scale;
  }

  /////////////////////
  // col [width-BLURSIZE, width]
  for (int x = width - BLUR_SIZE; x < width; x++) {
    t += in[width - 1]; // padding with in[w-1]
    t -= in[x - BLUR_SIZE - 1];
    out[x] = t * scale;
  }
}

__device__ void blur_y(float *out, float *in, int width, int height) {
  //int r = BLUR_SIZE;
  float scale = 1.0 / (float)((BLUR_SIZE << 1) + 1); // 1/(2r+1)

  /////////////////////
  // row [0, BLUR_SIZE]
  float t;
  t = in[0] * BLUR_SIZE; // padding with in[0]

  // accumulate BLUR_SIZE pixels to in[0]'s top
  // and average out
  for (int y = 0; y < (BLUR_SIZE + 1); y++) {
    t += in[y + width];
  }
  out[0] = t * scale;

  // take care of 0<y<BLUR_SIZE+1
  for (int y = 1; y < (BLUR_SIZE + 1); y++) {
    t += in[(y + BLUR_SIZE) * width];
    t -= in[0];
    out[y * width] = t * scale;
  }

  /////////////////////
  // row [BLUR_SIZE+1, width-BLURSIZE]
  for (int y = (BLUR_SIZE + 1); y < height - BLUR_SIZE; y++) {
    t += in[(y + BLUR_SIZE) * width];
    t -= in[(y - BLUR_SIZE) * width - width];
    out[y * width] = t * scale;
  }

  /////////////////////
  // row [width-BLURSIZE, width]
  for (int y = height - BLUR_SIZE; y < height; y++) {
    t += in[(height - 1) * width]; // padding with in[h-1]
    t -= in[(y - BLUR_SIZE) * width - width];
    out[y * width] = t * scale;
  }
}

__global__ void blurKernel_x(float *out, float *in, int width, int height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int idx_b = blockIdx.x;
  int idx_t = threadIdx.x;
  //__shared__ float SMEM_X[X_NTHREADS][X_NTHREADS * TILE_SIZE_X * sizeof(float)];
  if (y < height)
    blur_x(&out[y * width], &in[y * width], width, height);
}

__global__ void blurKernel_y(float *out, float *in, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  //__shared__ float SMEM_Y[Y_NTHREADS * TILE_SIZE_Y * sizeof(float)][Y_NTHREADS];
  if (x < width)
    blur_y(&out[x], &in[x], width, height);
}

///////////////////////////////////////////////////////

int main(int argc, char *argv[]) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  char *inputImageFile;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  float *deviceInputImageData;
  float *deviceOutputImageData;
  float *deviceTempImageData;
  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  inputImage = wbImport(inputImageFile);

  // The input image is in grayscale, so the number of channels is 1
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  printf("imageWidth, imageHeight = (%d, %d)\n", imageWidth, imageHeight);
  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);

  // Get host input and output image data
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);

  // Start timer
  timespec timer = tic();

  ///////////////////////////////////////////////////////
  // Allocate cuda memory for device input and ouput image data
  cudaHostAlloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float), 
             cudaHostAllocDefault);
  cudaHostAlloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float), 
             cudaHostAllocDefault);
  cudaHostAlloc((void **)&deviceTempImageData,
             imageWidth * imageHeight * sizeof(float), 
             cudaHostAllocDefault);

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);

  // Call your GPU kernel 10 times
  for (int i = 0; i < 10; i++) {
    //printf("Iteration %d\n", i);
    // horizontal pass
    {
      int nthreads = X_NTHREADS;
      dim3 threadsPerBlock(nthreads);
      dim3 blocksPerGrid(imageHeight / nthreads + 1);
      //printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n",
           //blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
           //threadsPerBlock.y);
      blurKernel_x<<<blocksPerGrid, threadsPerBlock>>>(
          deviceTempImageData, deviceInputImageData, imageWidth, imageHeight);
    }
     //vertical pass
    {
      int nthreads = Y_NTHREADS;
      dim3 threadsPerBlock(nthreads);
      dim3 blocksPerGrid(imageWidth / nthreads + 1);
      //printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n",
           //blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
           //threadsPerBlock.y);
      blurKernel_y<<<blocksPerGrid, threadsPerBlock>>> (
          deviceOutputImageData, deviceTempImageData, imageWidth, imageHeight);
    }
  }

  // Transfer data from GPU to CPU
  cudaMemcpy(hostOutputImageData, deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float), cudaMemcpyDeviceToHost);

  // Stop and print timer
  toc(&timer, "GPU execution time (including data transfer) in seconds");

  // Check the correctness of your solution
  wbSolution(args, outputImage);

  cudaFreeHost(deviceInputImageData);
  cudaFreeHost(deviceOutputImageData);
  cudaFreeHost(deviceTempImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}

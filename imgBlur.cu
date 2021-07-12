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
#define nthreads 64

///////////////////////////////////////////////////////
__device__ void blur_x(float *out, float *in, int width, int height) {
  int r = BLUR_SIZE;
  float scale = 1.0 / (float)((r << 1) + 1); // 1/(2r+1)

  /////////////////////
  // col [0, BLUR_SIZE]
  float t;
  t = in[0] * r; // padding with in[0]

  // accumulate BLUR_SIZE pixels to in[0]'s right
  // and average out
  for (int x = 0; x < (r + 1); x++) {
    t += in[x];
  }
  out[0] = t * scale;

  // take care of 0<x<BLUR_SIZE+1
  for (int x = 1; x < (r + 1); x++) {
    t += in[x + r];
    t -= in[0];  
    out[x] = t * scale;
  }

  /////////////////////
  // col [BLUR_SIZE+1, width-BLURSIZE]
  for (int x = (r + 1); x < width - r; x++) {
    t += in[x + r];
    t -= in[x - r - 1];
    out[x] = t * scale;
  }

  /////////////////////
  // col [width-BLURSIZE, width]
  for (int x = width - r; x < width; x++) {
    t += in[width - 1]; // padding with in[w-1]
    t -= in[x - r - 1];
    out[x] = t * scale;
  }
}

__device__ void blur_y(float *out, float *in, int width, int height) {
  int r = BLUR_SIZE;
  float scale = 1.0 / (float)((r << 1) + 1); // 1/(2r+1)

  /////////////////////
  // row [0, BLUR_SIZE]
  float t;
  t = in[0] * r; // padding with in[0]

  // accumulate BLUR_SIZE pixels to in[0]'s top
  // and average out
  for (int y = 0; y < (r + 1); y++) {
    t += in[y + width];
  }
  out[0] = t * scale;

  // take care of 0<y<BLUR_SIZE+1
  for (int y = 1; y < (r + 1); y++) {
    t += in[(y + r) * width];
    t -= in[0];
    out[y * width] = t * scale;
  }

  /////////////////////
  // row [BLUR_SIZE+1, width-BLURSIZE]
  for (int y = (r + 1); y < height - r; y++) {
    t += in[(y + r) * width];
    t -= in[(y - r) * width - width];
    out[y * width] = t * scale;
  }

  /////////////////////
  // row [width-BLURSIZE, width]
  for (int y = height - r; y < height; y++) {
    t += in[(height - 1) * width]; // padding with in[h-1]
    t -= in[(y - r) * width - width];
    out[y * width] = t * scale;
  }
}

__global__ void blurKernel_x(float *out, float *in, int width, int height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  // accessing by row -> insufficient
  blur_x(&out[y * width], &in[y * width], width, height);
}

__global__ void blurKernel_y(float *out, float *in, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
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
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceTempImageData,
             imageWidth * imageHeight * sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);

  // Call your GPU kernel 10 times
  for (int i = 0; i < 1; i++) {
    dim3 threadsPerBlock(nthreads);
    // horizontal pass
    {
      dim3 blocksPerGrid(imageHeight / nthreads + 1);
      blurKernel_x<<<blocksPerGrid, threadsPerBlock>>>(
          deviceTempImageData, deviceInputImageData, imageWidth, imageHeight);
    }
     //vertical pass
    {
      dim3 blocksPerGrid(imageWidth / nthreads + 1);
      blurKernel_y<<<blocksPerGrid, threadsPerBlock>>>(
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

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);

  return 0;
}

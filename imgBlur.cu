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

#define BLUR_SIZE 10
#define TILE_W 8
#define TILE_H 16
#define BLOCK_W (TILE_W + (2 * BLUR_SIZE))
#define BLOCK_H (TILE_H + (2 * BLUR_SIZE))
#define FILTER_W (BLUR_SIZE * 2 + 1)
#define FILTER_H (BLUR_SIZE * 2 + 1)
#define USE_SHARED_MEM

///////////////////////////////////////////////////////
//@@ INSERT YOUR CODE HERE
#ifndef USE_SHARED_MEM
__global__ void blurKernel(float *out, float *in, int width, int height) {
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Col < width && Row < height) {
    float pixVal = 0;
    float pixels = 0;
    for (int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow) {
      for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol) {
        int curRow = Row + blurRow;
        int curCol = Col + blurCol;
        if (curRow > -1 && curRow < height && curCol > -1 && curCol < width) {
          pixVal += in[curRow * width + curCol];
          pixels++;
        }
      }
    }
    out[Row * width + Col] = (pixVal / pixels);
  }
#else
__global__ void blurKernel(float *out, float *in, int width, int height) {
  int Col = blockIdx.x * TILE_W + threadIdx.x - BLUR_SIZE;
  int Row = blockIdx.y * TILE_H + threadIdx.y - BLUR_SIZE;
  int idx = Row * width + Col;

  __shared__ float smem[BLOCK_W][BLOCK_H];
  int in_bound = Col >= 0 && Row >= 0 && Col <= width && Row <= height; 
  smem[threadIdx.x][threadIdx.y] = in_bound ? in[idx] : 0;
  __syncthreads();

  // some of the loading threads don't participate in compute
  if (threadIdx.x >= BLUR_SIZE && threadIdx.x < BLOCK_W - BLUR_SIZE &&
      threadIdx.y >= BLUR_SIZE && threadIdx.y < BLOCK_H - BLUR_SIZE) {
    float pixVal = 0;
    int pixels = 0;
    for (int x = -BLUR_SIZE; x < BLUR_SIZE + 1; x++) {
      for (int y = -BLUR_SIZE; y < BLUR_SIZE + 1; y++) {
        pixVal += smem[threadIdx.x + x][threadIdx.y + y];
        pixels++;
      }
    }
    out[Row * width + Col] = (pixVal / pixels);
  }
#endif
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
  //@@ INSERT AND UPDATE YOUR CODE HERE

  // Allocate cuda memory for device input and ouput image data
  cudaMalloc((void **)&deviceInputImageData,
             imageWidth * imageHeight * sizeof(float));
  cudaMalloc((void **)&deviceOutputImageData,
             imageWidth * imageHeight * sizeof(float));

  // Transfer data from CPU to GPU
  cudaMemcpy(deviceInputImageData, hostInputImageData,
             imageWidth * imageHeight * sizeof(float), cudaMemcpyHostToDevice);

#ifndef USE_SHARED_MEM
  printf("NOT using shared memory\n");
  dim3 threadsPerBlock(TILE_W, TILE_H);
  dim3 blocksPerGrid(imageWidth / TILE_W, imageHeight / TILE_H);
#else
  printf("using shared memory\n");
  dim3 threadsPerBlock(BLOCK_W, BLOCK_H);
  dim3 blocksPerGrid(imageWidth / TILE_W + 1, imageHeight / TILE_H + 1);
#endif
  // Call your GPU kernel 10 times
  for (int i = 0; i < 1; i++)
    printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
           threadsPerBlock.y);
  blurKernel<<<blocksPerGrid, threadsPerBlock>>>(
      deviceOutputImageData, deviceInputImageData, imageWidth, imageHeight);

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

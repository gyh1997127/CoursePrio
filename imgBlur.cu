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
#define X_NTHREADS 16
#define HORIZONTAL_PASS_THREAD 32
#define Y_NTHREADS 135
#define FILTER_SIZE 1.0 / (float)((BLUR_SIZE << 1) + 1)
///////////////////////////////////////////////////////
__device__ void blur_x(int x, float *out, float *in, int width, int height) {
  // initial sum for different scenarios
  float sum = 0.0;
  // 0 <= x < Blur_size+1
  if (x < (BLUR_SIZE + 1)) {
    int padd_count = BLUR_SIZE - x;
    sum = in[0] * padd_count;
    for (int col = x; col < (BLUR_SIZE * 2 + 1 - padd_count); col++) {
      sum += in[col];
    }
    out[x] = sum * FILTER_SIZE;
  }
  // Blur_size+1 <= x < width - blur_size
  else if (x >= (BLUR_SIZE + 1) && x < (width - BLUR_SIZE)) {
    for (int col = (x - BLUR_SIZE); col < (x + BLUR_SIZE + 1); col++) {
      sum += in[col];
    }
    out[x] = sum * FILTER_SIZE;
  }
  // width-blursize <= x < width
  else if (x >= (width - BLUR_SIZE)) {
    int padd_count = x - (width - BLUR_SIZE - 1);
    sum = in[width - 1] * padd_count;
    for (int col = x; col < (BLUR_SIZE * 2 + 1 - padd_count); col++) {
      sum += in[col];
    }
    out[x] = sum * FILTER_SIZE;
  }

  for (int i = 1; i < width/HORIZONTAL_PASS_THREAD + 1; i++) {
    int col = x + i;
    if (col > width) return;
    if (col < (BLUR_SIZE + 1)) {
      sum += in[col];
      sum -= in[0];
      out[col] = sum * FILTER_SIZE;
    }
    if (col >= (BLUR_SIZE + 1) && (col < width - BLUR_SIZE)) {
      sum += in[col + BLUR_SIZE];
      sum -= in[col - BLUR_SIZE - 1];
      out[col] = sum * FILTER_SIZE;
    }
    if (col >= (width - BLUR_SIZE) && col < width) {
      sum += in[width - 1];
      sum -= in[col - BLUR_SIZE - 1];
      out[col] = sum * FILTER_SIZE;
    }
  }
}

__device__ void blur_y(float *out, float *in, int width, int height) {

  /////////////////////
  // row [0, BLUR_SIZE]
  float sum;
  sum = in[0] * BLUR_SIZE; // padding with in[0]

  // accumulate BLUR_SIZE pixels to in[0]'s top
  // and average out
  for (int row = 0; row < (BLUR_SIZE + 1); row++) {
    sum += in[row + width];
  }
  out[0] = sum * FILTER_SIZE;

  // take care of 0<row<BLUR_SIZE+1
  for (int row = 1; row < (BLUR_SIZE + 1); row++) {
    sum += in[(row + BLUR_SIZE) * width];
    sum -= in[0];
    out[row * width] = sum * FILTER_SIZE;
  }

  /////////////////////
  // row [BLUR_SIZE+1, width-BLURSIZE]
  for (int row = (BLUR_SIZE + 1); row < height - BLUR_SIZE; row++) {
    sum += in[(row + BLUR_SIZE) * width];
    sum -= in[(row - BLUR_SIZE) * width - width];
    out[row * width] = sum * FILTER_SIZE;
  }

  /////////////////////
  // row [width-BLURSIZE, width]
  for (int row = height - BLUR_SIZE; row < height; row++) {
    sum += in[(height - 1) * width]; // padding with in[h-1]
    sum -= in[(row - BLUR_SIZE) * width - width];
    out[row * width] = sum * FILTER_SIZE;
  }
}

__global__ void blurKernel_x(float *out, float *in, int width, int height) {
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int pixel_per_thread = (width)/HORIZONTAL_PASS_THREAD;// + 1;
  if (pixel_per_thread * Col < width && Row < height) {
    int start_idx_x = Col * pixel_per_thread; 
    blur_x(Col * pixel_per_thread, &out[Row * width + Col],
           &in[Row * width + Col], width, height);
  }
}

__global__ void blurKernel_y(float *out, float *in, int width, int height) {
  int Col = blockIdx.x * blockDim.x + threadIdx.x;
  if (Col < width)
    blur_y(&out[Col], &in[Col], width, height);
}

void blurKernel(float *out, float *in, float *temp, int imageWidth,
                int imageHeight) {
  // horizontal pass
  {
    int nthreads = X_NTHREADS;
    dim3 threadsPerBlock(HORIZONTAL_PASS_THREAD, nthreads);
    dim3 blocksPerGrid(imageWidth/HORIZONTAL_PASS_THREAD + 1, imageHeight / nthreads);
    printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
           threadsPerBlock.y);
    blurKernel_x<<<blocksPerGrid, threadsPerBlock>>>(temp, in, imageWidth,
                                                     imageHeight);
  }
  // vertical pass
  {
    int nthreads = Y_NTHREADS;
    dim3 threadsPerBlock(nthreads);
    dim3 blocksPerGrid(imageWidth / nthreads );
//    printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n",
//           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
//           threadsPerBlock.y);
    blurKernel_y<<<blocksPerGrid, threadsPerBlock>>>(out, temp, imageWidth,
                                                     imageHeight);
  }
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

  // The input image is in grayFILTER_SIZE, so the number of channels is 1
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  printf("imageWidth, imageHeight = (%d, %d)\n", imageWidth, imageHeight);
  // Since the image is monochromatic, it only contains only one channel
  outputImage = wbImage_new(imageWidth, imageHeight, 1);
  
  // Use pinned memory on host
  cudaHostAlloc((void **)&hostInputImageData, imageWidth * imageHeight * sizeof(float), cudaHostAllocDefault);
  cudaHostAlloc((void **)&hostOutputImageData, imageWidth * imageHeight * sizeof(float), cudaHostAllocDefault);

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
    blurKernel(deviceOutputImageData, deviceInputImageData, deviceTempImageData,
               imageWidth, imageHeight);
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

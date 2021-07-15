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
#define checkCudaErrors(err)           __checkCudaErrors (err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError err, const char *file, const int line)
{
    if (cudaSuccess != err)
    {
        fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",
                file, line, (int)err, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

///////////////////////////////////////////////////////
cudaChannelFormatDesc channelDesc =
  cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
void bind_texture(float *in, int width, int height, cudaArray_t cuArray, cudaTextureObject_t *texObj) {

  // the width in bytes of the 2D array pointed
  size_t pitch = width * sizeof(float);

  cudaMemcpy2DToArray(cuArray, 0, 0, in, pitch, width, height,
                      cudaMemcpyHostToDevice);

  // Specify texture
  struct cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;

  // Specify texture object parameters
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  //texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaCreateTextureObject(texObj, &resDesc, &texDesc, NULL);
}

__device__ void blur_x(float *out, float *in, int width, int height) {
  float scale = 1.0 / (float)((BLUR_SIZE << 1) + 1); // 1/(2r+1)

  /////////////////////
  // col [0, BLUR_SIZE]
  float sum;
  sum = in[0] * BLUR_SIZE; // padding with in[0]

  // accumulate BLUR_SIZE pixels to in[0]'s right
  // and average out
  for (int x = 0; x < (BLUR_SIZE + 1); x++) {
    sum += in[x];
  }
  out[0] = sum * scale;

  // take care of 0<x<BLUR_SIZE+1
  for (int x = 1; x < (BLUR_SIZE + 1); x++) {
    sum += in[x + BLUR_SIZE];
    sum -= in[0];  
    out[x] = sum * scale;
  }

  /////////////////////
  // col [BLUR_SIZE+1, width-BLURSIZE]
  for (int x = (BLUR_SIZE + 1); x < width - BLUR_SIZE; x++) {
    sum += in[x + BLUR_SIZE];
    sum -= in[x - BLUR_SIZE - 1];
    out[x] = sum * scale;
  }

  /////////////////////
  // col [width-BLURSIZE, width]
  for (int x = width - BLUR_SIZE; x < width; x++) {
    sum += in[width - 1]; // padding with in[w-1]
    sum -= in[x - BLUR_SIZE - 1];
    out[x] = sum * scale;
  }
}

__device__ void blur_y(float *out, float *in, int width, int height) {
  float scale = 1.0 / (float)((BLUR_SIZE << 1) + 1); // 1/(2r+1)

  /////////////////////
  // row [0, BLUR_SIZE]
  float sum;
  sum = in[0] * BLUR_SIZE; // padding with in[0]

  // accumulate BLUR_SIZE pixels to in[0]'s top
  // and average out
  for (int y = 0; y < (BLUR_SIZE + 1); y++) {
    sum += in[y + width];
  }
  out[0] = sum * scale;

  // take care of 0<y<BLUR_SIZE+1
  for (int y = 1; y < (BLUR_SIZE + 1); y++) {
    sum += in[(y + BLUR_SIZE) * width];
    sum -= in[0];
    out[y * width] = sum * scale;
  }

  /////////////////////
  // row [BLUR_SIZE+1, width-BLURSIZE]
  for (int y = (BLUR_SIZE + 1); y < height - BLUR_SIZE; y++) {
    sum += in[(y + BLUR_SIZE) * width];
    sum -= in[(y - BLUR_SIZE) * width - width];
    out[y * width] = sum * scale;
  }

  /////////////////////
  // row [width-BLURSIZE, width]
  for (int y = height - BLUR_SIZE; y < height; y++) {
    sum += in[(height - 1) * width]; // padding with in[h-1]
    sum -= in[(y - BLUR_SIZE) * width - width];
    out[y * width] = sum * scale;
  }
}

__global__ void blurKernel_x(float *out, float *in, int width, int height) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y < height)
    blur_x(&out[y * width], &in[y * width], width, height);
}

__global__ void blurKernel_x_TMEM(float *out, int width, int height, cudaTextureObject_t texObj) {
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y < height) {
    float scale = 1.0 / (float)((BLUR_SIZE << 1) + 1); // 1/(2r+1)
    float sum;

    // in[0]
    for (int x = -BLUR_SIZE; x < BLUR_SIZE; x++)
      sum += tex2D<float>(texObj, x, y);
    out[y * width + 0] = scale * sum;

    for (int x = 1; x < width; x++) {
      sum += tex2D<float>(texObj, x + BLUR_SIZE, y);
      sum -= tex2D<float>(texObj, x - BLUR_SIZE - 1, y);
      out[y * width + x] = scale * sum;
    }
  }
}

__global__ void blurKernel_y(float *out, float *in, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
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
  cudaArray_t cuArray;

  cudaTextureObject_t texObj = 0;
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

  // bind to texture mem
  cudaMallocArray(&cuArray, &channelDesc, imageWidth, imageHeight);
  bind_texture(hostInputImageData, imageWidth, imageHeight, cuArray, &texObj);

  // Call your GPU kernel 10 times
  for (int i = 0; i < 1; i++) {
    //printf("Iteration %d\n", i);
    // horizontal pass
    {
      int nthreads = X_NTHREADS;
      dim3 threadsPerBlock(nthreads);
      dim3 blocksPerGrid(imageHeight / nthreads + 1);
      printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
           threadsPerBlock.y);
      //blurKernel_x<<<blocksPerGrid, threadsPerBlock>>>(
          //deviceTempImageData, deviceInputImageData, imageWidth, imageHeight);
      blurKernel_x_TMEM<<<blocksPerGrid, threadsPerBlock>>> (deviceTempImageData, imageWidth, imageHeight, texObj);
    }
     //vertical pass
    {
      int nthreads = Y_NTHREADS;
      dim3 threadsPerBlock(nthreads);
      dim3 blocksPerGrid(imageWidth / nthreads + 1);
      printf("CUDA kernel launch with [%d %d] blocks of [%d %d] threads\n",
           blocksPerGrid.x, blocksPerGrid.y, threadsPerBlock.x,
           threadsPerBlock.y);
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

  cudaFree(deviceInputImageData);
  cudaFree(deviceOutputImageData);
  cudaFree(deviceTempImageData);
  cudaFreeArray(cuArray);
  // Destroy texture object
  cudaDestroyTextureObject(texObj);

  wbImage_delete(outputImage);
  wbImage_delete(inputImage);


  return 0;
}

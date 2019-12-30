#include "iradonT.h"

__host__ __device__ __inline__
static void iradonT_calc(float width, float center, float x, float y, float t, float* orig, float* dest)
{
  float u = (x-center)*cos(t) - (y-0.5*width)*sin(t) + center;
  if(0 <= u && u < width){
    dest[(int)(y*width + x)] += orig[(int)u];
  }
}

void iradonT_cpu(int width, int n_angles, float center, float* angles, float* orig, float* dest)
{
#pragma omp parallel for
  for(int y = 0; y < width; y++){
    for(int x = 0; x < width; x++){
      dest[x + y*width] = 0.0;
      for(int i = 0; i < n_angles; i++){
        iradonT_calc((float)width, center, (float)x, (float)y, angles[i], &orig[i*width], dest);
      }
    }
  }
}

__global__
static void iradonT_gpu_calc(int width, int n_angles, float center, float* angles, float* orig, float* dest)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = blockIdx.y;
  if(x < width){
    dest[x + y*width] = 0.0;
    for(int i = 0; i < n_angles; i++){
      iradonT_calc((float)width, center, (float)x, (float)y, angles[i], &orig[i*width], dest);
    }
  }
}

void iradonT_gpu(int width, int n_angles, float center, float* d_angles, float* d_orig, float* d_dest)
{
  int block = 256;
  dim3 grid((width + block - 1)/block, width);
  iradonT_gpu_calc<<<grid, block>>>(width, n_angles, center, d_angles, d_orig, d_dest);
}

/*
__host__ __device__ __inline__
static void diagRadonT_calc(float width, float center, float x, float y, float t, float *dest)
{
  float u = (x-center)*cos(t) - (y-0.5*width)*sin(t) + center;
  if(0.0 <= u && u < width){
    *dest = *dest + 1.0;
  }
}

void diagRadonT_cpu(int width, int n_angles, float center, float* angles, float* dest)
{
  for(int y = 0; y < width; y++){
    for(int x = 0; x < width; x++){
      dest[x + y*width] = 0.0;
      for(int i = 0; i < n_angles; i++){
        diagRadonT_calc((float)width, center, (float)x, (float)y, angles[i], &dest[x + y*width]);
      }
    }
  }
}

__global__
static void diagRadonT_gpu_calc(int width, int n_angles, float center, float* angles, float* dest)
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = blockIdx.y;
  if( x < width ){
    int id = x * y*width;
    dest[id] = 0.0;
    for(int i = 0; i < n_angles; i++){
      diagRadonT_calc((float)width, center, (float)x, (float)y, angles[i], &dest[x + y*width]);
    }
  }
}

float* diagRadonT_gpu(int width, int n_angles, float center, float* d_angles)
{
  float* d_dest;
  cudaMalloc(&d_dest, sizeof(float)*width*width);

  int block = 256;
  dim3 grid((width + block - 1)/block, width);
  diagRadonT_gpu_calc<<<grid, block>>>(width, n_angles, center, d_angles, d_dest);

  return d_dest;
}
*/


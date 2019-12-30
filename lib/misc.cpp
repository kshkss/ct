#include <cstdio>
#include <iostream>
#include <cstring>
#include <complex>
#include <fftw3.h>

#include "misc.h"

static int logical_size(int k){
  double r = ceil(log2((double)k));
  int n = (int)pow(2.0, r);

  //printf("logical size: %d\n", n);
  return n;
}

static void init_ramlak_filter(int width, int logical_size, double *dest)
{
  double a = 1.0/((double)logical_size);
  int length = 2 * (logical_size/2 + 1);
  memset(dest, 0, sizeof(double)*length/2);

#pragma opm parallel for
  for(int k = 0; k <= width/2; k++){
    dest[k] = a*(double)(k);
  }
}

void ramlak(int width, int n_angles, float* sino, float *dest)
{
  int n = logical_size(2*width);
  int length = 2 * (n/2 + 1);
  //double *filter = new double[length];
  double *filter = new double[length/2];
  //double spec[length * n_angles];
  //double *spec = new double[length * n_angles];
  double *spec = fftw_alloc_real(length * n_angles);

  init_ramlak_filter(width, n, filter);

  memset(spec, 0, sizeof(double)*length*n_angles);
#pragma omp parallel for
  for(int k = 0; k < n_angles; k++){
    for(int i = 0; i < width; i++){
      spec[i + k*length] = (double)sino[i + k*width];
    }
  }

  auto p1 = fftw_plan_many_dft_r2c(1, &n, n_angles, spec, nullptr, 1, length, reinterpret_cast<fftw_complex*>(spec), nullptr, 1, length/2, FFTW_ESTIMATE);
  fftw_execute(p1);
  fftw_destroy_plan(p1);

  auto *cspec = reinterpret_cast<std::complex<double>*>(spec);
#pragma omp parallel for
  for(int k = 0; k < n_angles; k++){
    for(int i = 0; i < length/2; i++){
      cspec[i + k*length/2] *= filter[i];
    }
  }

  auto p2 = fftw_plan_many_dft_c2r(1, &n, n_angles, reinterpret_cast<fftw_complex*>(spec), nullptr, 1, length/2, spec, nullptr, 1, length, FFTW_ESTIMATE);
  fftw_execute(p2);
  fftw_destroy_plan(p2);

  double r = M_PI / ((double)n * (double)n_angles);
#pragma omp parallel for
  for(int k = 0; k < n_angles; k++){
    for(int i = 0; i < width; i++){
      dest[i + k*width] = (float)(r * spec[i + k*length]);
    }
  }
  fftw_free(spec);
}

void complement(int width, int n_angles, char* mask, float *sino)
{
#pragma omp parallel for
  for(int i = 0; i < n_angles; i++){
    float left = 0.0;
    int count = 0;
    for(int j = 0; j < width; j++){
      int k = j + i * width;
      count += 1;
      if(mask[k] == 0){
        if(count > 1){
          float right = sino[k];
          for(int l = 1; l < count; l++){
            float r = (float)l / (float)count;
            sino[k-l] = r * left + (1.0 - r) * right;
          }
        }
        left = sino[k];
        count = 0;
      }
    }
    count += 1;
    if(count > 1){
      int k = (i + 1) * width;
      float right = 0.0;
      for(int l = 1; l < count; l++){
        float r = (float)l / (float)count;
        sino[k-l] = r * left + (1.0 - r) * right;
      }
    }
  }
}


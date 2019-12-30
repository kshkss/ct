#include <cstring>
#include <cstdio>
#include <cmath>
#include "device_ptr.h"
#include "misc.h"
#include "iradonT.h"

extern "C"{
	void fbp(float *sino, float *ct, int width, int n_angles, float *angles, float center);
}

void fbp(float *sino, float *ct, int width, int n_angles, float *angles, float center)
{
  float* orig = new float[width * n_angles];
  float* dest = ct;

  //memcpy(orig, sino, sizeof(float)*width*angles);
  ramlak(width, n_angles, sino, orig);

#define GPU
#ifndef GPU
  iradonT_cpu(width, n_angles, center, angles, orig, dest);
#else
  device_ptr<float> t(n_angles);
  t.copyToDevice(angles);

  device_ptr<float> x(width*width);
  x.setZero();

  device_ptr<float> b(width*n_angles);
  b.copyToDevice(orig);

  iradonT_gpu(width, n_angles, center, t.ptr, b.ptr, x.ptr);

  x.copyToHost(dest);
#endif

}

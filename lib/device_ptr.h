#ifndef __DEVICE_PTR_H__
#define __DEVICE_PTR_H__

#include <cstdio>
#include <cassert>
#include <cuda_runtime_api.h>

template<class T>
class device_ptr{
  device_ptr(device_ptr<T>& t);
  device_ptr<T> operator=(device_ptr<T>& t);

  public:
  const int size;
  T *ptr;

  device_ptr(int n):size(n), ptr(nullptr){
    auto err = cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T)*n);
    if(err != cudaSuccess){
      fprintf(stderr, "%s at device_ptr<T>(int).\nCheck the amount of memory usage.\n", cudaGetErrorName(err));
      assert(err == cudaSuccess);
    }
    //fprintf(stderr, "allocate device pointer: %p\n", ptr);
  }

  ~device_ptr(){
    auto err = cudaFree(ptr);
    //fprintf(stderr, "free device pointer: %p\n", ptr);
    if(err != cudaSuccess){
      fprintf(stderr, "%s at ~device_ptr<T>()\n", cudaGetErrorName(err));
      assert(err == cudaSuccess);
    }
  }

  device_ptr(device_ptr<T>&& t):size(t.size){
    ptr = t.ptr;
    t.ptr = nullptr;
  }

  void swap(device_ptr<T>& other_ptr){
    T* tmp = other_ptr.ptr;
    other_ptr.ptr = ptr;
    ptr = tmp;
  }
  
  void copyFrom(const device_ptr<T>& t){
    auto err = cudaMemcpy(ptr, t.ptr, sizeof(T)*size, cudaMemcpyDeviceToDevice);
    if(err != cudaSuccess){
      fprintf(stderr, "%s at copyFrom(const device_ptr<T>&)\n", cudaGetErrorName(err));
      assert(err == cudaSuccess);
    }
  }

  void setZero(){
    auto err = cudaMemset(ptr, 0, sizeof(T)*size);
    if(err != cudaSuccess){
      fprintf(stderr, "%s at setZero()\n", cudaGetErrorName(err));
      assert(err == cudaSuccess);
    }
  }

  void copyToDevice(const T *host_ptr){
    auto err = cudaMemcpy(ptr, host_ptr, sizeof(T)*size, cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
      fprintf(stderr, "%s at copyToDevice(const T*)\n", cudaGetErrorName(err));
      assert(err == cudaSuccess);
    }
  }

  void copyToHost(T *host_ptr){
    auto err = cudaMemcpy(host_ptr, ptr, sizeof(T)*size, cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
      //fprintf(stderr, "device pointer: %p\n", ptr);
      fprintf(stderr, "%s at copyToHost(T*)\n", cudaGetErrorName(err));
      assert(err == cudaSuccess);
    }
  }
};

#endif

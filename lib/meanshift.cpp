#include <cstdint>
#include <cstdio>
#include <cmath>

extern "C"{
  void histogram(uint8_t *vs, double *freq, double *offset, double *size, int *embed);
  void meanshift(uint8_t *vs, double *weight, double *offset, double *size, int *embed);
}

template<class T>
void histogram_epanechnikov(uint8_t *vs, T *freq, T *offset, T *size, int *embed)
{
  T cx = (offset[0] + 0.5*size[0]);
  T cy = (offset[1] + 0.5*size[1]);
  T offset_x = 0;
  T offset_y = 0;

  for(int y = (int)offset[1]; y <= (int)(offset[1] + size[1]); y++){
    for(int x = (int)offset[0]; x <= (int)(offset[0] + size[0]); x++){
      T dx = 2 * (static_cast<T>(x) - cx) / size[0];
      T dy = 2 * (static_cast<T>(y) - cy) / size[1];
      T weight = dx*dx + dy*dy;

      int k = y * embed[0] + x;
      freq[vs[k]] += fmax(1 - weight, 0.0);
    }
  }
}

template<class T>
void meanshift_epanechnikov(uint8_t *vs, T *weight, T *offset, T *size, int *embed)
{
  T cx = (offset[0] + 0.5*size[0]);
  T cy = (offset[1] + 0.5*size[1]);
  T wtot = 0;
  T offset_x = 0;
  T offset_y = 0;


  for(int y = (int)offset[1]; y <= (int)(offset[1] + size[1]); y++){
    for(int x = (int)offset[0]; x <= (int)(offset[0] + size[0]); x++){
      int k = y * embed[0] + x;
      T w = weight[vs[k]];

      T dx = 2 * (static_cast<T>(x) - cx) / size[0];
      T dy = 2 * (static_cast<T>(y) - cy) / size[1];
      if( dx*dx + dy*dy <= 1){
        wtot += w;
        offset_x += w*dx;
        offset_y += w*dy;
      }
    }
  }
  //printf("Total weight = %f\n", wtot);

  offset[0] += offset_x/wtot;
  offset[1] += offset_y/wtot;
}

template<class T>
void histogram_gauss(uint8_t *vs, T *freq, T *offset, T *size, int *embed)
{
  T cx = (offset[0] + 0.5*size[0]);
  T cy = (offset[1] + 0.5*size[1]);
  T offset_x = 0;
  T offset_y = 0;

  for(int y = (int)offset[1]; y <= (int)(offset[1] + size[1]); y++){
    for(int x = (int)offset[0]; x <= (int)(offset[0] + size[0]); x++){
  //for(int y = 0; y < embed[1]; y++){
    //for(int x = 0; x < embed[0]; x++){
      T dx = 2 * (static_cast<T>(x) - cx) / size[0];
      T dy = 2 * (static_cast<T>(y) - cy) / size[1];
      T weight = dx*dx + dy*dy;

      int k = y * embed[0] + x;
      freq[vs[k]] += exp(-0.5*weight * 16);
    }
  }
}

template<class T>
void meanshift_gauss(uint8_t *vs, T *weight, T *offset, T *size, int *embed)
{
  T cx = (offset[0] + 0.5*size[0]);
  T cy = (offset[1] + 0.5*size[1]);
  T wtot = 0;
  T offset_x = 0;
  T offset_y = 0;


  for(int y = (int)offset[1]; y <= (int)(offset[1] + size[1]); y++){
    for(int x = (int)offset[0]; x <= (int)(offset[0] + size[0]); x++){
  //for(int y = 0; y < embed[1]; y++){
    //for(int x = 0; x < embed[0]; x++){
      int k = y * embed[0] + x;
      T dx = 2 * (static_cast<T>(x) - cx) / size[0];
      T dy = 2 * (static_cast<T>(y) - cy) / size[1];
      T w = weight[vs[k]] * exp(-0.5*(dx*dx + dy*dy) * 16);

      wtot += w;
      offset_x += w*dx;
      offset_y += w*dy;
    }
  }
  //printf("Total weight = %f\n", wtot);

  offset[0] += offset_x/wtot;
  offset[1] += offset_y/wtot;
}

void histogram(uint8_t *vs, double *freq, double *offset, double *size, int *embed)
{
  //histogram_epanechnikov<double>(vs, freq, offset, size, embed);
  histogram_gauss<double>(vs, freq, offset, size, embed);
}

void meanshift(uint8_t *vs, double *weight, double *offset, double *size, int *embed)
{
  //meanshift_epanechnikov<double>(vs, weight, offset, size, embed);
  meanshift_gauss<double>(vs, weight, offset, size, embed);
}

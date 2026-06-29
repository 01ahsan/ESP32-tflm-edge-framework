#ifndef FEATURES_H
#define FEATURES_H

#include <cstdint>
#include "config.h"
#include "scaler.h"

float mean_val(const float* x, int n);
float std_val (const float* x, int n, float m);
float min_val (const float* x, int n);
float max_val (const float* x, int n);

inline float standardize(float v, int ch) {
  return (v - g_scaler.mean[ch]) / (g_scaler.std[ch] + 1e-9f);
}

inline int8_t quantize_to_int8(float z, float scale, int zero_point) {
  int32_t q = static_cast<int32_t>(z / scale) + zero_point;
  if (q >  127) q =  127;
  if (q < -128) q = -128;
  return static_cast<int8_t>(q);
}

void extract_features(const float* ppg, float* out,
                      float fs = 50.0f, bool scaled = true);

#endif  // FEATURES_H

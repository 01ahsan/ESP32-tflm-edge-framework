#ifndef SCALER_H
#define SCALER_H

struct Scaler {
  float mean[12];
  float std[12];
  float output_mean;
  float output_std;
};

extern Scaler g_scaler;

#endif  // SCALER_H

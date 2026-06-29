// Auto-generated feature extraction. Mirrors Python logic.
#include "features.h"
#include <cmath>
#include <algorithm>
#include <cfloat>

float mean_val(const float* x, int n) {
  if (n <= 0) return 0.0f;
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += x[i];
  return s / n;
}

float std_val(const float* x, int n, float m) {
  if (n <= 1) return 0.0f;
  float s = 0.0f;
  for (int i = 0; i < n; ++i) s += (x[i] - m) * (x[i] - m);
  return sqrtf(s / n);
}

float min_val(const float* x, int n) {
  float v = FLT_MAX;
  for (int i = 0; i < n; ++i) if (x[i] < v) v = x[i];
  return v;
}

float max_val(const float* x, int n) {
  float v = -FLT_MAX;
  for (int i = 0; i < n; ++i) if (x[i] > v) v = x[i];
  return v;
}

static int find_peaks_c(const float* x, int n, int* peaks,
                        int max_peaks, int min_dist, float min_prom) {
  int np = 0;
  for (int i = 1; i < n - 1 && np < max_peaks; ++i) {
    if (x[i] <= x[i - 1] || x[i] <= x[i + 1]) continue;
    if (np > 0 && (i - peaks[np - 1]) < min_dist) {
      if (x[i] > x[peaks[np - 1]]) peaks[np - 1] = i;
      continue;
    }
    float lmin = x[i], rmin = x[i];
    for (int j = i - 1; j >= 0; --j) if (x[j] < lmin) lmin = x[j];
    for (int j = i + 1; j < n; ++j) if (x[j] < rmin) rmin = x[j];
    if (x[i] - std::max(lmin, rmin) >= min_prom) peaks[np++] = i;
  }
  return np;
}

void extract_features(const float* ppg, float* out, float fs, bool scaled) {
  for (int i = 0; i < kFeatures; ++i) out[i] = 0.0f;

  const int N      = kSeqLen;
  float ppg_min    = min_val(ppg, N);
  float ppg_max    = max_val(ppg, N);
  float ppg_rng    = ppg_max - ppg_min + 1e-9f;
  int   min_dist   = static_cast<int>(0.4f * fs);
  float min_prom   = 0.3f * ppg_rng;

  static int peaks[64], valleys[64];
  int np = find_peaks_c(ppg, N, peaks, 64, min_dist, min_prom);

  int nv = 0;
  for (int i = 0; i < np - 1 && nv < 64; ++i) {
    int best = peaks[i];
    for (int j = peaks[i] + 1; j < peaks[i + 1]; ++j)
      if (ppg[j] < ppg[best]) best = j;
    valleys[nv++] = best;
  }

  if (np >= 2) {
    float s = 0, sq = 0;
    for (int i = 1; i < np; ++i) s += (peaks[i] - peaks[i - 1]);
    float ibi = s / (np - 1) / fs;
    out[0] = 60.0f / ibi;
    for (int i = 1; i < np; ++i) {
      float d = (peaks[i] - peaks[i - 1]) / fs - ibi;
      sq += d * d;
    }
    out[1] = sqrtf(sq / (np - 1)) * 1000.0f;
  } else {
    out[0] = 75.0f;
  }

  if (np > 0 && nv > 0) {
    int k = std::min(np, nv);
    float s = 0, sq = 0;
    for (int i = 0; i < k; ++i) s += ppg[peaks[i]] - ppg[valleys[i]];
    float am = s / k;
    out[2] = am;
    for (int i = 0; i < k; ++i) {
      float d = ppg[peaks[i]] - ppg[valleys[i]] - am;
      sq += d * d;
    }
    out[3] = sqrtf(sq / k);
  } else {
    out[2] = ppg_rng;
  }

  float rt = 0, dt = 0;
  int rn = 0, dn = 0;
  for (int pi = 0; pi < np; ++pi) {
    for (int vi = nv - 1; vi >= 0; --vi) {
      if (valleys[vi] < peaks[pi]) {
        rt += (peaks[pi] - valleys[vi]) / fs;
        rn++;
        break;
      }
    }
    for (int vi = 0; vi < nv; ++vi) {
      if (valleys[vi] > peaks[pi]) {
        dt += (valleys[vi] - peaks[pi]) / fs;
        dn++;
        break;
      }
    }
  }
  out[4] = rn ? rt / rn : 0.0f;
  out[5] = dn ? dt / dn : 0.0f;

  float pw50 = 0;
  int pw50n = 0;
  for (int pi = 0; pi < np; ++pi) {
    int foot = -1, nxt = -1;
    for (int vi = nv - 1; vi >= 0; --vi) if (valleys[vi] < peaks[pi]) { foot = valleys[vi]; break; }
    for (int vi = 0; vi < nv; ++vi) if (valleys[vi] > peaks[pi]) { nxt = valleys[vi]; break; }
    if (foot < 0 || nxt < 0) continue;
    float th = ppg[foot] + 0.5f * (ppg[peaks[pi]] - ppg[foot]);
    int first = -1, last = -1;
    for (int j = foot; j <= nxt; ++j) if (ppg[j] >= th) { if (first < 0) first = j; last = j; }
    if (first >= 0 && last > first) { pw50 += (last - first) / fs; pw50n++; }
  }
  out[6] = pw50n ? pw50 / pw50n : 0.0f;

  float auc = 0;
  int aucn = 0;
  for (int pi = 0; pi < np; ++pi) {
    int foot = -1, nxt = -1;
    for (int vi = nv - 1; vi >= 0; --vi) if (valleys[vi] < peaks[pi]) { foot = valleys[vi]; break; }
    for (int vi = 0; vi < nv; ++vi) if (valleys[vi] > peaks[pi]) { nxt = valleys[vi]; break; }
    if (foot < 0 || nxt < 0) continue;
    float base = ppg[foot], a = 0;
    for (int j = foot; j < nxt; ++j) a += 0.5f * ((ppg[j] - base) + (ppg[j + 1] - base));
    auc += a;
    aucn++;
  }
  out[7] = aucn ? auc / aucn : 0.0f;

  out[8] = mean_val(ppg, N);
  out[9] = std_val(ppg, N, out[8]);

  float vs = 0;
  for (int i = 0; i < N - 1; ++i) vs += fabsf(ppg[i + 1] - ppg[i]);
  out[10] = vs / (N - 1);

  {
    float apg[kSeqLen - 2];
    for (int i = 0; i < N - 2; ++i) apg[i] = (ppg[i + 2] - ppg[i + 1]) - (ppg[i + 1] - ppg[i]);
    float apg_std = std_val(apg, N - 2, mean_val(apg, N - 2));
    float apg_p = apg_std * 0.3f;
    int ap[32];
    int nap = find_peaks_c(apg, N - 2, ap, 32, 1, apg_p);
    float neg[kSeqLen - 2];
    for (int i = 0; i < N - 2; ++i) neg[i] = -apg[i];
    int an[32];
    int nan_ = find_peaks_c(neg, N - 2, an, 32, 1, apg_p);
    if (nap > 0 && nan_ > 0) {
      float a = apg[ap[0]];
      for (int i = 0; i < nan_; ++i) {
        if (an[i] > ap[0]) {
          out[11] = apg[an[i]] / (a + 1e-9f);
          break;
        }
      }
    }
  }

  for (int i = 0; i < kFeatures; ++i) if (!std::isfinite(out[i])) out[i] = 0.0f;
  if (scaled) for (int i = 0; i < kFeatures; ++i) out[i] = standardize(out[i], i);
}

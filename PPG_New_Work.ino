#include <Arduino.h>
#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_GFX.h>
#include <Adafruit_SH110X.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "esp_heap_caps.h"
#include "config.h"
#include "model.h"
#include "scaler.h"
#include "features.h"

MAX30105 particleSensor;
Adafruit_SH1106G display(128, 64, &Wire, -1);

uint8_t* tensor_arena = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input  = nullptr;
TfLiteTensor* output = nullptr;

float* ppg_window  = nullptr;
float* feature_vec = nullptr;
float* flat_input  = nullptr;
float* ppg_norm    = nullptr;

int      sample_index   = 0;
uint32_t next_sample_us = 0;

float latest_pred     = NAN;
float last_latency_ms = 0.0f;
uint32_t window_count = 0;

float clamp_glucose(float g) {
  if (g < kGlucoseMin) return kGlucoseMin;
  if (g > kGlucoseMax) return kGlucoseMax;
  return g;
}

void init_display() {
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!display.begin(0x3C, true)) {
    Serial.println("ERROR: OLED not found.");
    while (1) delay(1000);
  }
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println("Initializing...");
  display.display();
}

void show_status_screen(const char* line1, const char* line2 = nullptr) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);
  display.setTextSize(1);
  display.setCursor(0, 0);
  display.println(line1);
  if (line2) display.println(line2);
  display.display();
}

void update_display(float pred, float lat_ms) {
  display.clearDisplay();
  display.setTextColor(SH110X_WHITE);

  display.setTextSize(1);
  display.setCursor(0, 0);
  display.print("Glucose  win#");
  display.print(window_count);

  display.setTextSize(2);
  display.setCursor(0, 12);
  if (!isnan(pred)) {
    display.print(pred, 1);
    display.print(" mg");
  } else {
    display.print("--- mg");
  }

  display.setTextSize(1);
  display.setCursor(0, 38);
  display.print("Infer: ");
  display.print(lat_ms, 1);
  display.print(" ms");

  display.setCursor(0, 50);
  display.print("Heap: ");
  display.print(ESP.getFreeHeap() / 1024);
  display.print(" KB free");

  display.display();
}

void init_sensor() {
  Wire.begin(SDA_PIN, SCL_PIN);
  if (!particleSensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("ERROR: MAX30102 not found.");
    show_status_screen("MAX30102 Error");
    while (1) delay(1000);
  }
  particleSensor.setup(60, 4, 2, 100, 411, 4096);
  particleSensor.setPulseAmplitudeRed(0x0A);
  particleSensor.setPulseAmplitudeGreen(0);
  Serial.println("MAX30102 OK.");
}

void init_model() {
  tensor_arena = (uint8_t*)heap_caps_malloc(
      kTensorArenaSize, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT);
  if (!tensor_arena) {
    Serial.println("ERROR: tensor_arena alloc failed.");
    while (1) delay(1000);
  }

  const tflite::Model* model = tflite::GetModel(model_tflite);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema mismatch.");
    while (1) delay(1000);
  }

  static tflite::MicroMutableOpResolver<26> resolver;
  resolver.AddFullyConnected();
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAveragePool2D();
  resolver.AddMaxPool2D();
  resolver.AddRelu();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddAdd();
  resolver.AddMul();
  resolver.AddQuantize();
  resolver.AddDequantize();
  resolver.AddMean();
  resolver.AddPad();
  resolver.AddConcatenation();
  resolver.AddExpandDims();
  resolver.AddReduceMax();
  resolver.AddStridedSlice();
  resolver.AddTranspose();

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors failed.");
    while (1) delay(1000);
  }

  input  = interpreter->input(0);
  output = interpreter->output(0);

  Serial.println("Model OK.");
}

void normalize_ppg(const float* raw, float* out, int n) {
  float lo = raw[0], hi = raw[0];
  for (int i = 1; i < n; ++i) {
    if (raw[i] < lo) lo = raw[i];
    if (raw[i] > hi) hi = raw[i];
  }
  float rng = (hi - lo < 1e-6f) ? 1e-6f : hi - lo;
  for (int i = 0; i < n; ++i)
    out[i] = 2.0f * (raw[i] - lo) / rng - 1.0f;
}

void build_model_input() {
  normalize_ppg(ppg_window, ppg_norm, kSeqLen);
  for (int i = 0; i < kSeqLen; ++i)
    flat_input[i] = ppg_norm[i];
  extract_features(ppg_norm, feature_vec, 100.0f, true);
  for (int i = 0; i < kFeatures; ++i)
    flat_input[kSeqLen + i] = feature_vec[i];
}

float run_inference(float& lat_ms_out) {
  build_model_input();

  if (input->type == kTfLiteFloat32) {
    for (int i = 0; i < kInputLen; ++i)
      input->data.f[i] = flat_input[i];
  } else if (input->type == kTfLiteInt8) {
    float sc = input->params.scale;
    int   zp = input->params.zero_point;
    for (int i = 0; i < kInputLen; ++i) {
      int32_t q = (int32_t)roundf(flat_input[i] / sc) + zp;
      if (q < -128) q = -128;
      if (q >  127) q =  127;
      input->data.int8[i] = (int8_t)q;
    }
  } else {
    lat_ms_out = 0;
    return NAN;
  }

  uint32_t t0 = micros();
  TfLiteStatus st = interpreter->Invoke();
  uint32_t t1 = micros();

  lat_ms_out = (float)(t1 - t0) / 1000.0f;

  if (st != kTfLiteOk) {
    Serial.println("[ERROR] Invoke() failed.");
    return NAN;
  }

  float g = NAN;
  if (output->type == kTfLiteFloat32) {
    g = output->data.f[0];
  } else if (output->type == kTfLiteInt8) {
    g = (output->data.int8[0] - output->params.zero_point)
        * output->params.scale;
  }

  return clamp_glucose(g);
}

void mainTask(void *pvParameters);

void setup() {
  Serial.begin(115200);
  delay(1200);

  ppg_window  = (float*)heap_caps_malloc(sizeof(float) * kSeqLen,   MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  feature_vec = (float*)heap_caps_malloc(sizeof(float) * kFeatures, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  flat_input  = (float*)heap_caps_malloc(sizeof(float) * kInputLen, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
  ppg_norm    = (float*)heap_caps_malloc(sizeof(float) * kSeqLen,   MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);

  if (!ppg_window || !feature_vec || !flat_input || !ppg_norm) {
    Serial.println("ERROR: PSRAM buffer alloc failed.");
    while (1) delay(1000);
  }

  Serial.println("PPG inference starting.");

  init_display();
  init_sensor();
  init_model();

  Serial.println("System ready.");
  Serial.println("Waiting for finger.");

  show_status_screen("Place finger", "on sensor...");

  sample_index   = 0;
  window_count   = 0;
  next_sample_us = micros();
  xTaskCreatePinnedToCore(
      mainTask,
      "mainTask",
      24576,
      NULL,
      1,
      NULL,
      1
  );
}

void mainTask(void *pvParameters) {
  while (true) {
    if ((int32_t)(micros() - next_sample_us) < 0) {
      vTaskDelay(1);
      continue;
    }
    next_sample_us += 10000;

    long ir = particleSensor.getIR();

    if (ir < 50000) {
      if (sample_index > 0) {
        Serial.println("[WARN] Finger lost. Window reset.");
        sample_index = 0;
      }
      show_status_screen("No finger", "Place on sensor");
      next_sample_us = micros();
      vTaskDelay(1);
      continue;
    }

    if (ir < 10000 || ir > 200000)
      ir = (sample_index > 0) ? (long)ppg_window[sample_index - 1] : 50000L;

    ppg_window[sample_index] = (float)ir;
    sample_index++;

    if (sample_index % 100 == 0) {
      char prog[32];
      snprintf(prog, sizeof(prog), "%d / %d samples", sample_index, kSeqLen);
      show_status_screen("Collecting...", prog);
    }

    if (sample_index < kSeqLen) {
      continue;
    }

    float mn = ppg_window[0], mx = ppg_window[0];
    for (int i = 1; i < kSeqLen; i++) {
      if (ppg_window[i] < mn) mn = ppg_window[i];
      if (ppg_window[i] > mx) mx = ppg_window[i];
    }
    float signal_range = mx - mn;

    if (signal_range < 5000) {
      Serial.println("[WARN] Signal too flat. Adjust finger pressure.");
      show_status_screen("Bad signal", "Adjust finger");
      sample_index   = 0;
      next_sample_us = micros();
      continue;
    }

    show_status_screen("Running model...", "please wait");

    float lat_ms = 0.0f;
    float pred   = run_inference(lat_ms);

    window_count++;
    latest_pred     = pred;
    last_latency_ms = lat_ms;

    Serial.print("[RESULT] Window ");
    Serial.print(window_count);
    Serial.print(" | Inference ");
    Serial.print(lat_ms, 3);
    Serial.print(" ms | Prediction ");
    if (!isnan(pred)) {
      Serial.print(pred, 2);
      Serial.println(" mg/dL");
    } else {
      Serial.println("NAN");
    }

    if (!isnan(pred)) update_display(pred, lat_ms);
    else              show_status_screen("Infer failed", "Check model");

    sample_index   = 0;
    next_sample_us = micros();
  }
}

void loop() {
  vTaskDelay(portMAX_DELAY);
}

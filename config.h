#ifndef CONFIG_H
#define CONFIG_H

#define SDA_PIN 8
#define SCL_PIN 9

constexpr int kSeqLen          = 1000;      // 100 Hz * 10 sec
constexpr int kFeatures        = 12;
constexpr int kInputLen        = kSeqLen + kFeatures;
constexpr int kTensorArenaSize = 228 * 1024;

const int sample_interval_ms   = 10;        // 100 Hz

constexpr float kGlucoseMin    = 40.0f;
constexpr float kGlucoseMax    = 400.0f;

const char* WIFI_SSID     = "YOUR_WIFI";
const char* WIFI_PASSWORD = "YOUR_PASSWORD";
const char* MQTT_BROKER   = "YOUR_BROKER_IP";

// Activation used during training: ReLU6 (builtin)
// All TFLite ops are builtin only - MCU-safe

#endif  // CONFIG_H

# ESP32-S3 TinyML PPG Project


<img width="1378" height="1719" alt="pic" src="https://github.com/user-attachments/assets/f190c538-fa70-4a96-9508-f103c915817d" />


This repository contains the first working release of an ESP32-S3 TinyML PPG inference demo.

## What is included

- `PPG_New_Work.ino` - working firmware entry point
- `config.h` - hardware and deployment configuration
- `features.cpp` / `features.h` - PPG feature extraction
- `scaler.cpp` / `scaler.h` - feature scaling helpers
- `model.h` - TensorFlow Lite Micro model declaration
- `sample_model.cc` - embedded sample model data for immediate testing
- `coordinator.py` - MQTT federation coordinator with Robbins-Monro updates

## Setup

1. Update these placeholders in `config.h`:
   - `WIFI_SSID`
   - `WIFI_PASSWORD`
   - `MQTT_BROKER`

2. Update this placeholder in `coordinator.py`:
   - `BROKER_HOST`

3. Install the Python coordinator dependency:

```bash
pip install -r requirements.txt
```

4. Flash the firmware to an ESP32-S3 board with the required Arduino libraries installed.

## Notes

- Generated coordinator logs such as `coordinator_log.jsonl` are ignored by Git.
- The deployment uses the embedded model source, so the original `.tflite` export is not included here.
- Replace `sample_model.cc` with your own converted model source when deploying a new trained model.
- The firmware is kept as a working Arduino sketch for this first release. A future cleanup can split it into `firmware/main.cpp`, `mqtt`, `sensor`, `display`, `inference`, `calibration`, and `storage` modules.

## Current capabilities

- ESP32-S3 TinyML deployment
- TensorFlow Lite Micro inference
- MAX30102 PPG acquisition
- Feature extraction and scaling
- Calibration-ready configuration
- MQTT federation coordinator
- Robbins-Monro population prior updates

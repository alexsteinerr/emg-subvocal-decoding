#ifndef DEVICE_HPP
#define DEVICE_HPP

#include <WebSocketsClient.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <HTTPUpdate.h>

#include <driver/i2s.h>
#include <SPI.h>

#include <Preferences.h>

#include <queue>
#include <vector>
#include <string>
#include <iostream>
#include <algorithm>

#include <EEPROM.h>


// TensorFlow Lite Micro 
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include <esp_heap_caps.h>

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>

#include "helper/audio/WAVFileReader.h"
#include "helper/audio/I2SOutput.h"
#include "helper/audio/AudioConfig.h" 
#include "helper/audio/AudioPlayer.h"

#include "helper/camera/CameraHandler.h"

#include "helper/wake_word/model.h"
#include "ADS1299.hpp"

using namespace std;

// PSRAM Allocator
template <typename T>
struct PSRAMAllocator {
    typedef T value_type;
    PSRAMAllocator() = default;
    template <typename U> PSRAMAllocator(const PSRAMAllocator<U>&) {}
    T* allocate(std::size_t n) {
        return (T*)heap_caps_malloc(n * sizeof(T), MALLOC_CAP_SPIRAM);
    }
    void deallocate(T* p, std::size_t n) {
        heap_caps_free(p);
    }
};

namespace tflite {
    template <unsigned int tOpCount>
    class MicroMutableOpResolver;
    class ErrorReporter;
    class Model;
    class MicroInterpreter;
} 

struct TfLiteTensor;

struct AudioQueueItem {
    std::vector<uint8_t> data;
    AudioOutputTarget target;
};

class Device : public BLECharacteristicCallbacks {
  public:
    WebSocketsClient client;
    AudioPlayer audioPlayer;
    CameraHandler camera;
    Preferences preferences;
    ADS1299* ads;

    std::string AUTH_KEY;

    int volume = 100;

    enum current_state {
        connected,
        not_connected,
        waiting_for_trigger, 
        processing_eeg,    
        streaming_eeg,
        ota      
    };

    current_state current_state = not_connected;
    bool is_streaming = false;

    static constexpr size_t DMA_BUF_LEN = 512;   // samples per buffer
    static constexpr size_t DMA_BUF_COUNT = 8;

  private:
    // TFLite components
    tflite::MicroMutableOpResolver<12> *m_resolver;
    tflite::ErrorReporter *m_error_reporter;
    const tflite::Model *m_model;
    tflite::MicroInterpreter *m_interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t *m_tensor_arena;

    const float maxBatteryVoltage = 4.2; 
    const float minBatteryVoltage = 3.0;

    std::string version = "1.0.0";

  public:
      // Setup Methods
      void setup_tf();
      void setup_ble();
      void setup_ads(); 

      // EMG Methods
      void stream_eeg_data();
      int16_t* get_eeg_sample();
      void wait_for_eeg_trigger(); 

      // Battery
      void check_battery();
  
      // Audio 
      void set_volume(string volume);

      // Internal helpers
      void write_i2s(i2s_port_t port, const int16_t* samples, size_t count);
      void convert_and_scale(const int16_t* input, size_t input_samples,
                        int16_t* output, size_t output_samples,
                        int gain, bool stereo_out);

      // Camera
      void setup_camera();
      void stream_camera_to_ws();

      // TFLite Prediction 
      bool predict(int16_t* input_buffer);

      // Connectivity / Utilities
      void connect_wifi(const char *ssid, const char *password);
      void disconnect_wifi();
      void invoke_error(const char* err);

      // BLE Handling
      void listen_ble();
      void send_ble(const std::string& payload);
      void process_ble_data(std::string data);
      void onWrite(BLECharacteristic *pCharacteristic);
      void close_ble();

      // Persistence
      void save_string(String key, const string value);
      String read_string(String key);

      // String Helpers
      vector<string> split(string s, string delimiter);
      String generate_uuid();

      void check_for_update();
      string get_version();
};

#endif
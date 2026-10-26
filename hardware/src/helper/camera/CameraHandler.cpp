#include "CameraHandler.h"
#include <esp_task_wdt.h>

CameraHandler::CameraHandler() {}

bool CameraHandler::begin() {
    Serial.println("[CAM] Initialising camera...");
    Serial.println("[WARN] PSRAM disabled in config to avoid IO48 bus collision.");

    // 1. Watchdog management
    esp_task_wdt_init(15, true);
    esp_task_wdt_add(nullptr);
    esp_task_wdt_reset();

    // 2. Mapping based on provided schematic (FPC-05F-24PH20)
    _config.pin_xclk     = 38;  // FPC Pin 3  (IO38)
    _config.pin_vsync    = 17;  // FPC Pin 4  (IO17)
    _config.pin_pclk     = 45;  // FPC Pin 5  (IO45)
    _config.pin_href     = 47;  // FPC Pin 6  (IO47)
    
    _config.pin_d0       = 37;  // FPC Pin 7  (IO37)
    _config.pin_d1       = 48;  // FPC Pin 8  (IO48) - THE CONFLICT PIN
    _config.pin_d2       = 1;   // FPC Pin 9  (IO1)
    _config.pin_d3       = 2;   // FPC Pin 10 (IO2)
    _config.pin_d4       = 21;  // FPC Pin 11 (IO21)
    _config.pin_d5       = 39;  // FPC Pin 12 (IO39)
    _config.pin_d6       = 40;  // FPC Pin 13 (IO40)
    _config.pin_d7       = 41;  // FPC Pin 14 (IO41)

    // SCCB (I2C) - Matching FPC Pins 16/17 labels
    _config.pin_sscb_sda = 16;  // FPC Pin 16 (IO16)
    _config.pin_sscb_scl = 14;  // FPC Pin 17 (IO14)

    _config.pin_pwdn     = -1;  // Not explicitly labeled on IO
    _config.pin_reset    = -1;  // Not explicitly labeled on IO

    // 3. Signal & Memory Settings
    _config.xclk_freq_hz = 20000000;       // 20MHz standard
    _config.ledc_timer   = LEDC_TIMER_0;
    _config.ledc_channel = LEDC_CHANNEL_0;
    _config.pixel_format = PIXFORMAT_JPEG; 
    _config.frame_size   = FRAMESIZE_QVGA; // 320x240
    _config.jpeg_quality = 12;
    _config.fb_count     = 1;
    
    // CRITICAL: Force the frame buffer into Internal RAM (DRAM)
    // Since PSRAM is disabled, allocating in PSRAM will cause a crash.
    _config.fb_location  = CAMERA_FB_IN_DRAM; 

    esp_task_wdt_reset();

    // 4. Initialize
    esp_err_t err = esp_camera_init(&_config);
    
    if (err != ESP_OK) {
        Serial.printf("[CAM] Init failed: 0x%x\n", err);
        // Error 0x105 = I2C not finding the sensor
        // Error 0x101 = Camera didn't send a signal (PCLK/VSYNC issue)
        return false;
    }

    Serial.println("[CAM] Camera Initialised Successfully in DRAM mode.");
    return true;
}

camera_fb_t* CameraHandler::capture() {
    return esp_camera_fb_get();
}

void CameraHandler::release(camera_fb_t* fb) {
    esp_camera_fb_return(fb);
}
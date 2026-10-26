#ifndef CAMERA_HANDLER_H
#define CAMERA_HANDLER_H

#include "esp_camera.h"
#include <Arduino.h>

#define PWDN_GPIO_NUM    -1
#define RESET_GPIO_NUM   -1
#define XCLK_GPIO_NUM    14
#define SIOD_GPIO_NUM    21
#define SIOC_GPIO_NUM    46

#define Y9_GPIO_NUM      38
#define Y8_GPIO_NUM      17
#define Y7_GPIO_NUM      45
#define Y6_GPIO_NUM      47
#define Y5_GPIO_NUM      37
#define Y4_GPIO_NUM      48
#define Y3_GPIO_NUM      1
#define Y2_GPIO_NUM      2
#define VSYNC_GPIO_NUM   39
#define HREF_GPIO_NUM    40
#define PCLK_GPIO_NUM    41

class CameraHandler {
public:
    CameraHandler();
    bool begin();
    camera_fb_t* capture();
    void release(camera_fb_t* fb);

private:
    camera_config_t _config;
};

#endif
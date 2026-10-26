#include "device.hpp"
#include <esp_task_wdt.h>

void Device::setup_camera() {
    if (!camera.begin()) {
        // Do NOT hang in an infinite loop – reboot after printing error
        Serial.println("FATAL: Camera initialisation failed – rebooting in 5 seconds...");
        delay(5000);
        esp_restart();
    }
}

void Device::stream_camera_to_ws() {
    camera_fb_t* fb = camera.capture();
    if (!fb) {
        Serial.println("Camera capture failed – skipping frame");
        // Feed watchdog to prevent reset if captures keep failing
        esp_task_wdt_reset();
        return;
    }

    client.sendBIN(fb->buf, fb->len);
    camera.release(fb);

    // Tiny delay to yield to other tasks (EEG, TFLite)
    delay(1);

    // Feed watchdog after every frame – prevents reset in long loops
    esp_task_wdt_reset();
}
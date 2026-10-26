#include "I2SOutput.h"
#include "AudioConfig.h"
#include <esp_heap_caps.h>

#define NUM_FRAMES      512   // smaller chunks to fit small buffers

static Frame_t* s_frames = nullptr;
static bool s_frames_allocated = false;

static void allocate_frames() {
    if (!s_frames_allocated) {
        s_frames = (Frame_t*)heap_caps_malloc(NUM_FRAMES * sizeof(Frame_t), MALLOC_CAP_SPIRAM);
        if (!s_frames) {
            s_frames = (Frame_t*)malloc(NUM_FRAMES * sizeof(Frame_t));
        }
        s_frames_allocated = true;
    }
}

I2SOutput::I2SOutput(i2s_port_t port) 
    : m_port(port), m_source(nullptr), m_taskHandle(nullptr), m_playing(false), m_shouldExit(false) {
    m_mutex = xSemaphoreCreateMutex();
    allocate_frames();
}

I2SOutput::~I2SOutput() {
    stopAndWait();
    if (m_mutex) {
        vSemaphoreDelete(m_mutex);
        m_mutex = nullptr;
    }
}

void I2SOutput::start(SampleSource* source) {
    if (!source) return;
    stopAndWait();
    xSemaphoreTake(m_mutex, portMAX_DELAY);
    m_source = source;
    m_playing = true;
    m_shouldExit = false;
    xSemaphoreGive(m_mutex);
    if (!m_taskHandle) {
        xTaskCreatePinnedToCore(taskWrapper, "i2s_out", 32768, this, 2, &m_taskHandle, 1);
    } else {
        Serial.println("[I2S] Task already exists, not recreating");
    }
}

void I2SOutput::stop() {
    xSemaphoreTake(m_mutex, portMAX_DELAY);
    m_shouldExit = true;
    m_playing = false;
    xSemaphoreGive(m_mutex);
}

void I2SOutput::stopAndWait() {
    stop();
    if (m_taskHandle) {
        for (int i = 0; i < 50; i++) {
            if (m_taskHandle == nullptr) break;
            delay(10);
        }
        if (m_taskHandle) {
            vTaskDelete(m_taskHandle);
            m_taskHandle = nullptr;
        }
    }
}

void I2SOutput::run() {
    uint32_t writeCount = 0;
    while (true) {
        xSemaphoreTake(m_mutex, portMAX_DELAY);
        bool play = m_playing;
        bool exit = m_shouldExit;
        SampleSource* src = m_source;
        xSemaphoreGive(m_mutex);

        if (exit) break;

        if (play && src) {
            src->getFrames(s_frames, NUM_FRAMES);
            size_t bytes_written = 0;
            esp_err_t err = i2s_write(m_port, (const char*)s_frames,
                                      NUM_FRAMES * sizeof(Frame_t),
                                      &bytes_written, portMAX_DELAY);
            writeCount++;
            if (err != ESP_OK) {
                Serial.printf("[I2S] Write error: %d\n", err);
            }
            if (src->isFinished()) {
                xSemaphoreTake(m_mutex, portMAX_DELAY);
                m_playing = false;
                xSemaphoreGive(m_mutex);
            }
        } else {
            delay(1);
        }
    }
    m_taskHandle = nullptr;
    vTaskDelete(NULL);
}

bool I2SOutput::isPlaying() const {
    bool ret;
    xSemaphoreTake(m_mutex, portMAX_DELAY);
    ret = m_playing && (m_taskHandle != nullptr);
    xSemaphoreGive(m_mutex);
    return ret;
}

void I2SOutput::taskWrapper(void* param) {
    static_cast<I2SOutput*>(param)->run();
}
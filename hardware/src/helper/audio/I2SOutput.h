#ifndef I2SOUTPUT_H
#define I2SOUTPUT_H

#include <Arduino.h>
#include "driver/i2s.h"
#include "SampleSource.h"
#include <freertos/semphr.h>

class I2SOutput {
public:
    I2SOutput(i2s_port_t port);
    ~I2SOutput();

    void start(SampleSource* source);
    void stop();            // non-blocking, signals task to exit
    void stopAndWait();     // blocks until task is fully deleted
    bool isPlaying() const;

    i2s_port_t getPort() const { return m_port; }

private:
    static void taskWrapper(void* param);
    void run();

    i2s_port_t m_port;
    SampleSource* m_source;
    TaskHandle_t m_taskHandle;
    volatile bool m_playing;
    volatile bool m_shouldExit;
    SemaphoreHandle_t m_mutex;   // protects m_source and m_playing
};

#endif
#ifndef AUDIOPLAYER_H
#define AUDIOPLAYER_H

#include <queue>
#include <vector>
#include <cstdint>
#include <utility>

#include "I2SOutput.h"
#include "WAVFileReader.h"
#include "BufferSampleSource.h"
#include "AudioConfig.h"

struct PlayBufferItem {
    uint8_t* data;        // will be freed after processing
    size_t len;
    float volume;
    AudioOutputTarget target;
};

class AudioPlayer {
public:
    AudioPlayer();
    ~AudioPlayer();

    void init();

    void playFile(const char* path, AudioOutputTarget target);

    // Play a raw PCM buffer (22050 Hz, mono, 16-bit)
    void playBuffer(const uint8_t* data, size_t len, AudioOutputTarget target, float volume = 1.0f);

    void waitForCompletion(AudioOutputTarget target);
    bool isPlaying(AudioOutputTarget target);
    void stop(AudioOutputTarget target);

    void setVolume(int percent);
    
    void resetStream(AudioOutputTarget target);
    void endStream(AudioOutputTarget target);

private:
    static void playTaskWrapper(void* param);
    void playTask();

    I2SOutput* m_outputBone;
    I2SOutput* m_outputSpeaker;
    BufferSampleSource* m_sourceBone;      // persistent source for bone
    BufferSampleSource* m_sourceSpeaker;   // persistent source for speaker
    int m_volumePercent;

    // Queue and task handles
    QueueHandle_t m_playQueue;
    TaskHandle_t  m_playTaskHandle;
    uint8_t* m_scratchOut;
    size_t   m_scratchOutCap;
};

#endif
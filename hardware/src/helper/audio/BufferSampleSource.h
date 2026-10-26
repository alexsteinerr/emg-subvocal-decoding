#ifndef BUFFERSAMPLESOURCE_H
#define BUFFERSAMPLESOURCE_H

#include "SampleSource.h"
#include "AudioConfig.h"
#include <Arduino.h>

class BufferSampleSource : public SampleSource {
public:
    BufferSampleSource();
    ~BufferSampleSource();

    void appendBuffer(const uint8_t* buffer, size_t size, float volume, bool stereo);
    void clear();

    void beginStream();
    void markEndOfStream();

    int sampleRate() override { return SAMPLE_RATE; }
    void getFrames(Frame_t* frames, int numFrames) override;
    bool isFinished() override { return false; }   // always streaming

private:
    static constexpr size_t MAX_CHUNKS = 128;
    static constexpr size_t MAX_CHUNK_SIZE = 2048;

    static constexpr int PRIME_MS = 20;
    static constexpr int REPRIME_MS = 10;

    static constexpr size_t PRIME_THRESHOLD_BYTES   = (size_t)SAMPLE_RATE * 4 * PRIME_MS / 1000;
    static constexpr size_t REPRIME_THRESHOLD_BYTES = (size_t)SAMPLE_RATE * 4 * REPRIME_MS / 1000;

    static constexpr int FADE_SAMPLES = 220; // ~5ms at 44.1kHz, avoids clicks at boundaries

    struct Chunk {
        uint8_t data[MAX_CHUNK_SIZE];
        size_t size;
        float volume;
        bool stereo;
        bool valid;
    };

    enum FadeState { FADE_NONE, FADE_IN, FADE_OUT };

    Chunk m_chunks[MAX_CHUNKS];
    int m_writeIndex;
    int m_readIndex;
    int m_count;

    // Current chunk being read
    uint8_t* m_readData;
    size_t   m_readSize;
    float    m_readVolume;
    bool     m_readStereo;
    size_t   m_readPos;

    // Priming / underrun handling
    volatile size_t m_bufferedBytes; 
    bool m_priming;       
    bool m_hasPlayedAnything;  
    volatile bool m_streamEnded;    

    FadeState m_fadeState;
    int m_fadeCounter;
    int16_t m_lastLeft, m_lastRight;   // last emitted sample, used as fade-out base

 
    volatile uint32_t m_underrunCount;
    unsigned long m_lastUnderrunLogMs;

    portMUX_TYPE m_mux;
};

#endif
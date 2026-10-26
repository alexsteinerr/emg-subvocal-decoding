#include "BufferSampleSource.h"
#include <cstring>

BufferSampleSource::BufferSampleSource()
    : m_writeIndex(0), m_readIndex(0), m_count(0),
      m_readData(nullptr), m_readSize(0), m_readVolume(1.0f),
      m_readStereo(true), m_readPos(0),
      m_bufferedBytes(0), m_priming(true), m_hasPlayedAnything(false),
      m_streamEnded(true), // no active stream yet
      m_fadeState(FADE_NONE), m_fadeCounter(0),
      m_lastLeft(0), m_lastRight(0) {
    m_mux = portMUX_INITIALIZER_UNLOCKED;
    clear();
}

BufferSampleSource::~BufferSampleSource() {
    clear();
}

void BufferSampleSource::appendBuffer(const uint8_t* buffer, size_t size,
                                      float volume, bool stereo) {
    if (!buffer || size == 0 || size > MAX_CHUNK_SIZE) {
        Serial.printf("[BSS] Invalid append: size=%u\n", (unsigned)size);
        return;
    }

    portENTER_CRITICAL(&m_mux);
    if (m_count >= MAX_CHUNKS) {
        portEXIT_CRITICAL(&m_mux);
        Serial.println("[BSS] Chunk ring full, dropping incoming chunk");
        return;
    }

    int idx = m_writeIndex;
    Chunk* chunk = &m_chunks[idx];
    memcpy(chunk->data, buffer, size);
    chunk->size = size;
    chunk->volume = volume;
    chunk->stereo = stereo;
    chunk->valid = true;
    m_writeIndex = (m_writeIndex + 1) % MAX_CHUNKS;
    m_count++;
    m_bufferedBytes += size;
    portEXIT_CRITICAL(&m_mux);
}

void BufferSampleSource::clear() {
    portENTER_CRITICAL(&m_mux);
    m_writeIndex = 0;
    m_readIndex = 0;
    m_count = 0;
    m_readPos = 0;
    m_readData = nullptr;
    m_readSize = 0;
    m_bufferedBytes = 0;
    for (int i = 0; i < MAX_CHUNKS; ++i) {
        m_chunks[i].valid = false;
    }
    portEXIT_CRITICAL(&m_mux);

    m_priming = true;
    m_hasPlayedAnything = false;
    m_fadeState = FADE_NONE;
    m_fadeCounter = 0;
    m_lastLeft = 0;
    m_lastRight = 0;
}

void BufferSampleSource::beginStream() {
    portENTER_CRITICAL(&m_mux);

    m_streamEnded = false;
    m_priming = true;

    portEXIT_CRITICAL(&m_mux);
}

void BufferSampleSource::markEndOfStream() {
    m_streamEnded = true;
}

void BufferSampleSource::getFrames(Frame_t* frames, int numFrames) {
    for (int i = 0; i < numFrames; ++i) {
        if (m_priming) {
            portENTER_CRITICAL(&m_mux);
            size_t buffered = m_bufferedBytes;
            portEXIT_CRITICAL(&m_mux);

            bool streamEnded = m_streamEnded;
            size_t threshold = m_hasPlayedAnything ? REPRIME_THRESHOLD_BYTES : PRIME_THRESHOLD_BYTES;

            bool readyToPlay;

            if (streamEnded) {
                readyToPlay = buffered > 0;
            } else {
                readyToPlay = buffered >= threshold;
            }

            if (readyToPlay) {
                m_priming = false;
                if (buffered > 0) {
                    m_fadeState = FADE_IN;
                    m_fadeCounter = FADE_SAMPLES;
                }
                // fall through and start actually playing this frame
            } else if (m_fadeState != FADE_OUT) {
                // Not yet enough data, and no fade-out tail to finish playing
                frames[i].left = 0;
                frames[i].right = 0;
                m_lastLeft = 0;
                m_lastRight = 0;
                continue;
            }
        }


        if (m_readData == nullptr || m_readPos >= m_readSize) {
            portENTER_CRITICAL(&m_mux);
            if (m_count > 0) {
                Chunk* chunk = &m_chunks[m_readIndex];
                if (chunk->valid) {
                    m_readData = chunk->data;
                    m_readSize = chunk->size;
                    m_readVolume = chunk->volume;
                    m_readStereo = chunk->stereo;
                    m_readPos = 0;
                    chunk->valid = false;
                    m_readIndex = (m_readIndex + 1) % MAX_CHUNKS;
                    m_count--;
                    m_bufferedBytes -= (chunk->size > m_bufferedBytes) ? m_bufferedBytes : chunk->size;
                } else {
                    m_readData = nullptr;
                    m_readSize = 0;
                }
            } else {
                if (m_readData != nullptr) {
                    m_fadeState = FADE_OUT;
                    m_fadeCounter = FADE_SAMPLES;
                }
                m_readData = nullptr;
                m_readSize = 0;
                if (!m_streamEnded) {
                    m_priming = true;
                }
            }
            portEXIT_CRITICAL(&m_mux);
        }

        int16_t leftSample = 0, rightSample = 0;
        bool haveSample = false;

        if (m_readData != nullptr && m_readSize != 0) {
            size_t bytesPerFrame = m_readStereo ? 4 : 2;
            if (m_readPos + bytesPerFrame <= m_readSize) {
                uint8_t* src = m_readData + m_readPos;
                if (m_readStereo) {
                    leftSample  = (int16_t)(src[0] | (src[1] << 8));
                    rightSample = (int16_t)(src[2] | (src[3] << 8));
                } else {
                    leftSample  = (int16_t)(src[0] | (src[1] << 8));
                    rightSample = leftSample;
                }
                m_readPos += bytesPerFrame;

                float vol = m_readVolume;
                if (vol > 1.0f) vol = 1.0f;
                if (vol < 0.0f) vol = 0.0f;
                vol *= 0.75f; // headroom

                leftSample  = (int16_t)constrain((int32_t)((int32_t)leftSample  * vol), -32768, 32767);
                rightSample = (int16_t)constrain((int32_t)((int32_t)rightSample * vol), -32768, 32767);
                haveSample = true;
                m_hasPlayedAnything = true;
            } else {
                // Incomplete frame – should not happen; skip to next chunk
                m_readData = nullptr;
                m_readSize = 0;
                --i; // retry
                continue;
            }
        }

        if (!haveSample) {
            leftSample = 0;
            rightSample = 0;
        }

        if (m_fadeState == FADE_IN) {
            float t = 1.0f - (float)m_fadeCounter / FADE_SAMPLES;
            leftSample  = (int16_t)(leftSample  * t);
            rightSample = (int16_t)(rightSample * t);
            if (--m_fadeCounter <= 0) m_fadeState = FADE_NONE;
        } else if (m_fadeState == FADE_OUT) {
            float t = (float)m_fadeCounter / FADE_SAMPLES;
            leftSample  = (int16_t)(m_lastLeft  * t);
            rightSample = (int16_t)(m_lastRight * t);
            if (--m_fadeCounter <= 0) m_fadeState = FADE_NONE;
        }

        m_lastLeft = leftSample;
        m_lastRight = rightSample;

        frames[i].left = leftSample;
        frames[i].right = rightSample;
    }
}
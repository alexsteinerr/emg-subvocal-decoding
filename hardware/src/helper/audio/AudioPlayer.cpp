#include "AudioPlayer.h"
#include "driver/i2s.h"
#include <LittleFS.h>
#include <esp_heap_caps.h>

static i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_16BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = I2S_DMA_BUF_COUNT,
    .dma_buf_len = I2S_DMA_BUF_LEN,
    .use_apll = false,
    .tx_desc_auto_clear = true,
    .fixed_mclk = 0
};

static constexpr size_t MAX_CHUNK_BYTES = 2048;
static constexpr size_t MAX_INPUT_CHUNK_BYTES = 8192 * 2; 
static constexpr size_t MAX_OUTPUT_CHUNK_BYTES = MAX_INPUT_CHUNK_BYTES * 4;

AudioPlayer::AudioPlayer()
    : m_outputBone(nullptr), m_outputSpeaker(nullptr),
      m_sourceBone(nullptr), m_sourceSpeaker(nullptr),
      m_volumePercent(100),
      m_playQueue(nullptr), m_playTaskHandle(nullptr),
      m_scratchOut(nullptr), m_scratchOutCap(0) {}

AudioPlayer::~AudioPlayer() {
    // Stop the play task
    if (m_playTaskHandle) {
        vTaskDelete(m_playTaskHandle);
        m_playTaskHandle = nullptr;
    }
    if (m_playQueue) {
        vQueueDelete(m_playQueue);
        m_playQueue = nullptr;
    }
    stop(TARGET_BONE);
    stop(TARGET_SPEAKER);
    if (m_outputBone) delete m_outputBone;
    if (m_outputSpeaker) delete m_outputSpeaker;
    if (m_sourceBone) delete m_sourceBone;
    if (m_sourceSpeaker) delete m_sourceSpeaker;
    if (m_scratchOut) {
        heap_caps_free(m_scratchOut);
        m_scratchOut = nullptr;
    }
}

void AudioPlayer::init() {
    // Bone is disabled
    // Speaker I2S
    i2s_pin_config_t pin_speaker = {
        .bck_io_num = PIN_SPEAKER_BCK,
        .ws_io_num = PIN_SPEAKER_WS,
        .data_out_num = PIN_SPEAKER_DOUT,
        .data_in_num = I2S_PIN_NO_CHANGE
    };
    if (i2s_driver_install(I2S_PORT_SPEAKER, &i2s_config, 0, NULL) != ESP_OK) {
        Serial.println("Failed to install speaker I2S");
        return;
    }
    i2s_set_pin(I2S_PORT_SPEAKER, &pin_speaker);
    i2s_set_sample_rates(I2S_PORT_SPEAKER, SAMPLE_RATE);
    m_outputSpeaker = new I2SOutput(I2S_PORT_SPEAKER);

    m_sourceSpeaker = new BufferSampleSource();

    // Start the speaker task with the buffer source
    m_outputSpeaker->start(m_sourceSpeaker);
    Serial.println("[AudioPlayer] Initialized speaker output with BufferSampleSource");

    m_scratchOutCap = MAX_OUTPUT_CHUNK_BYTES;
    m_scratchOut = (uint8_t*)heap_caps_malloc(m_scratchOutCap, MALLOC_CAP_SPIRAM);
    if (!m_scratchOut) {
        Serial.println("[AudioPlayer] Failed to allocate scratch buffer in PSRAM, falling back to internal RAM");
        m_scratchOut = (uint8_t*)malloc(m_scratchOutCap);
    }
    if (!m_scratchOut) {
        Serial.println("[AudioPlayer] CRITICAL: scratch buffer allocation failed entirely");
    }

    // Create queue for playBuffer items
    m_playQueue = xQueueCreate(20, sizeof(PlayBufferItem));
    if (!m_playQueue) {
        Serial.println("[AudioPlayer] Failed to create play queue");
        return;
    }

    // Create dedicated task for processing playBuffer requests.
    xTaskCreatePinnedToCore(playTaskWrapper, "playBufferTask", 8192, this, 2, &m_playTaskHandle, 0);
    if (!m_playTaskHandle) {
        Serial.println("[AudioPlayer] Failed to create play task");
    }
}

void AudioPlayer::playFile(const char* path, AudioOutputTarget target) {
    if (target == TARGET_BONE && m_outputBone == nullptr) {
        target = TARGET_SPEAKER;
    }
    I2SOutput* out = (target == TARGET_BONE) ? m_outputBone : m_outputSpeaker;
    if (!out) return;

    out->stopAndWait();

    WAVFileReader* reader = new WAVFileReader(LittleFS, path);
    if (!reader->isValid() || reader->sampleRate() != SAMPLE_RATE) {
        Serial.printf("Invalid WAV: %s\n", path);
        delete reader;
        BufferSampleSource* src = (target == TARGET_BONE) ? m_sourceBone : m_sourceSpeaker;
        out->start(src);
        return;
    }

    out->start(reader);
    waitForCompletion(target);
    delete reader;

    // Switch back to the buffer source
    BufferSampleSource* src = (target == TARGET_BONE) ? m_sourceBone : m_sourceSpeaker;
    Serial.printf("[AudioPlayer] Switching back to %s BufferSampleSource\n",
                  (target == TARGET_BONE) ? "bone" : "speaker");
    out->start(src);
}

void AudioPlayer::playBuffer(const uint8_t* data, size_t len, AudioOutputTarget target, float volume) {
    if (!data || len == 0) return;
    if (target == TARGET_BONE && m_outputBone == nullptr) {
        target = TARGET_SPEAKER;
    }
    // Only speaker is supported for streaming
    if (!m_playQueue) {
        Serial.println("[AudioPlayer] play queue not available");
        return;
    }

    if (len > MAX_INPUT_CHUNK_BYTES) {
        Serial.printf("[AudioPlayer] playBuffer: chunk too large (%u bytes), dropping\n", (unsigned)len);
        return;
    }


    uint8_t* copy = (uint8_t*)heap_caps_malloc(len, MALLOC_CAP_SPIRAM);
    if (!copy) {
        copy = (uint8_t*)malloc(len); // fallback if PSRAM is exhausted
    }
    if (!copy) {
        Serial.println("[AudioPlayer] playBuffer: allocation failed");
        return;
    }
    memcpy(copy, data, len);

    PlayBufferItem item;
    item.data = copy;
    item.len = len;
    item.volume = volume;
    item.target = target;

    if (xQueueSend(m_playQueue, &item, 0) != pdPASS) {
        heap_caps_free(copy);
        Serial.println("[AudioPlayer] playBuffer queue full, dropped chunk");
    }
}

bool AudioPlayer::isPlaying(AudioOutputTarget target) {
    if (target == TARGET_BONE && m_outputBone == nullptr) target = TARGET_SPEAKER;
    I2SOutput* out = (target == TARGET_BONE) ? m_outputBone : m_outputSpeaker;
    return out ? out->isPlaying() : false;
}

void AudioPlayer::waitForCompletion(AudioOutputTarget target) {
    while (isPlaying(target)) {
        delay(10);
    }
}

void AudioPlayer::stop(AudioOutputTarget target) {
    if (target == TARGET_BONE && m_outputBone == nullptr) target = TARGET_SPEAKER;
    I2SOutput* out = (target == TARGET_BONE) ? m_outputBone : m_outputSpeaker;
    if (out) out->stopAndWait();
    BufferSampleSource* src = (target == TARGET_BONE) ? m_sourceBone : m_sourceSpeaker;
    if (src) src->clear();

    // Clear any pending playBuffer items
    if (m_playQueue) {
        PlayBufferItem item;
        while (xQueueReceive(m_playQueue, &item, 0) == pdPASS) {
            heap_caps_free(item.data);
        }
    }

    if (out && src) out->start(src);
}

void AudioPlayer::setVolume(int percent) {
    m_volumePercent = constrain(percent, 0, 100);
}

void AudioPlayer::resetStream(AudioOutputTarget target) {
    if (target == TARGET_BONE && m_outputBone == nullptr) target = TARGET_SPEAKER;
    BufferSampleSource* src = (target == TARGET_BONE) ? m_sourceBone : m_sourceSpeaker;
    if (src) {
        src->clear();       // flush stale audio, re-arm priming
        src->beginStream();  // mark a real stream as active (full pre-buffer required)
    }
}

void AudioPlayer::endStream(AudioOutputTarget target) {
    if (target == TARGET_BONE && m_outputBone == nullptr) target = TARGET_SPEAKER;
    BufferSampleSource* src = (target == TARGET_BONE) ? m_sourceBone : m_sourceSpeaker;
    if (src) src->markEndOfStream(); // no more data coming; stop waiting on thresholds
}

void AudioPlayer::playTaskWrapper(void* param) {
    static_cast<AudioPlayer*>(param)->playTask();
}

void AudioPlayer::playTask() {
    PlayBufferItem item;
    while (true) {
        if (xQueueReceive(m_playQueue, &item, portMAX_DELAY) == pdPASS) {
            const uint8_t* inData = item.data;
            size_t inLen = item.len;
            float vol = item.volume;

            // Combine with global volume
            float globalVol = m_volumePercent / 100.0f;
            vol *= globalVol;
            if (vol > 1.0f) vol = 1.0f;
            if (vol < 0.0f) vol = 0.0f;

            // Input is mono 16-bit PCM. Each sample = 2 bytes.
            // Output is stereo 16-bit, with two stereo frames per input sample.
            // So output bytes = input bytes * 4 (because each sample produces 8 bytes,
            // but we process in blocks to keep latency low).
            const int16_t* inSamples = (const int16_t*)inData;
            size_t numInputSamples = inLen / 2;

            // Process in small blocks to avoid long blocking times
            const size_t BLOCK_SAMPLES = 256;          // 256 mono samples
            size_t processed = 0;

            while (processed < numInputSamples) {
                size_t blockSamples = (numInputSamples - processed) < BLOCK_SAMPLES ?
                                      (numInputSamples - processed) : BLOCK_SAMPLES;
                size_t blockOutBytes = blockSamples * 8;   // each sample → 8 bytes (2 stereo frames)

                // The scratch buffer is large enough (≥ MAX_OUTPUT_CHUNK_BYTES)
                // and blockOutBytes is well within that.
                if (blockOutBytes > m_scratchOutCap) {
                    Serial.printf("[AudioPlayer] blockOutBytes (%u) > scratch cap (%u), aborting\n",
                                  (unsigned)blockOutBytes, (unsigned)m_scratchOutCap);
                    break;
                }

                int16_t* out = (int16_t*)m_scratchOut;
                for (size_t i = 0; i < blockSamples; i++) {
                    int16_t sample = inSamples[processed + i];
                    // Write two stereo frames (left=right=sample)
                    out[4*i + 0] = sample;   // frame 1 left
                    out[4*i + 1] = sample;   // frame 1 right
                    out[4*i + 2] = sample;   // frame 2 left
                    out[4*i + 3] = sample;   // frame 2 right
                }

                // Append this converted block to the speaker buffer
                if (m_sourceSpeaker) {
                    m_sourceSpeaker->appendBuffer(m_scratchOut, blockOutBytes, vol, true);
                } else {
                    Serial.println("[AudioPlayer] playTask: m_sourceSpeaker is null");
                }

                processed += blockSamples;
            }

            // Free the original data copy
            heap_caps_free(item.data);
        }
    }
}
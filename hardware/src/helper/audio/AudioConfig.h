#ifndef AUDIO_CONFIG_H
#define AUDIO_CONFIG_H

#include <driver/i2s.h>

enum AudioOutputTarget {
    TARGET_BONE,
    TARGET_SPEAKER
};

#define SAMPLE_RATE 44100
#define BITS_PER_SAMPLE 16

// Bone conduction – disabled to save memory
#define I2S_PORT_BONE       I2S_NUM_0
#define PIN_BONE_BCK        8
#define PIN_BONE_WS         9
#define PIN_BONE_DOUT       10

// Loudspeaker – primary output
#define I2S_PORT_SPEAKER    I2S_NUM_1
#define PIN_SPEAKER_BCK     11
#define PIN_SPEAKER_WS      12  
#define PIN_SPEAKER_DOUT    13

// Larger DMA buffers to avoid underruns
#define I2S_DMA_BUF_COUNT   16
#define I2S_DMA_BUF_LEN     1024  

extern bool m_isPlaying[2];

#endif
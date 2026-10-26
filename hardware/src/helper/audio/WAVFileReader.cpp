#include "WAVFileReader.h"
#include <Arduino.h>

#pragma pack(push, 1)
struct RiffHeader {
    char riff[4];
    uint32_t totalSize;
    char wave[4];
};
struct FmtChunk {
    char id[4];
    uint32_t size;
    uint16_t audioFormat;
    uint16_t numChannels;
    uint32_t sampleRate;
    uint32_t byteRate;
    uint16_t blockAlign;
    uint16_t bitsPerSample;
};
struct Chunk {
    char id[4];
    uint32_t size;
};
#pragma pack(pop)

WAVFileReader::WAVFileReader(fs::FS& fs, const char* path)
    : m_sampleRate(0), m_numChannels(0), m_dataStart(0), m_dataBytes(0), m_readPos(0), m_valid(false) {
    if (!fs.exists(path)) {
        Serial.printf("WAV file not found: %s\n", path);
        return;
    }
    m_file = fs.open(path, "r");
    if (!m_file) return;

    // Read RIFF header
    RiffHeader riff;
    if (m_file.read((uint8_t*)&riff, sizeof(riff)) != sizeof(riff)) return;
    if (memcmp(riff.riff, "RIFF", 4) != 0 || memcmp(riff.wave, "WAVE", 4) != 0) return;

    // Find fmt chunk
    FmtChunk fmt;
    bool fmtFound = false;
    while (m_file.available()) {
        Chunk chunk;
        if (m_file.read((uint8_t*)&chunk, sizeof(chunk)) != sizeof(chunk)) break;
        if (memcmp(chunk.id, "fmt ", 4) == 0) {
            if (chunk.size >= sizeof(fmt) - 8) {
                if (m_file.read((uint8_t*)&fmt + 8, sizeof(fmt) - 8) != sizeof(fmt) - 8) break;
                fmtFound = true;
            }
            break;
        } else {
            m_file.seek(m_file.position() + chunk.size);
        }
    }
    if (!fmtFound) return;
    if (fmt.audioFormat != 1) { // PCM only
        Serial.printf("Unsupported audio format: %d\n", fmt.audioFormat);
        return;
    }
    if (fmt.bitsPerSample != 16) {
        Serial.printf("Unsupported bit depth: %d\n", fmt.bitsPerSample);
        return;
    }
    m_numChannels = fmt.numChannels;
    m_sampleRate = fmt.sampleRate;

    // Find data chunk
    while (m_file.available()) {
        Chunk chunk;
        if (m_file.read((uint8_t*)&chunk, sizeof(chunk)) != sizeof(chunk)) break;
        if (memcmp(chunk.id, "data", 4) == 0) {
            m_dataStart = m_file.position();
            m_dataBytes = chunk.size;
            m_valid = true;
            break;
        } else {
            m_file.seek(m_file.position() + chunk.size);
        }
    }

    if (!m_valid) {
        Serial.println("No data chunk found");
        m_file.close();
    } else {
        Serial.printf("[WAV] Loaded: %d Hz, %d channels, %d bytes\n", m_sampleRate, m_numChannels, m_dataBytes);
    }
}

WAVFileReader::~WAVFileReader() {
    if (m_file) m_file.close();
}

void WAVFileReader::getFrames(Frame_t* frames, int numFrames) {
    if (!m_valid || !m_file) {
        memset(frames, 0, numFrames * sizeof(Frame_t));
        return;
    }

    for (int i = 0; i < numFrames; i++) {
        if (m_readPos >= m_dataBytes) {
            memset(&frames[i], 0, (numFrames - i) * sizeof(Frame_t));
            m_valid = false; // mark finished
            return;
        }
        int16_t left = 0, right = 0;
        size_t bytesRead = 0;
        if (m_numChannels == 1) {
            bytesRead = m_file.read((uint8_t*)&left, 2);
            if (bytesRead != 2) {
                m_valid = false;
                memset(&frames[i], 0, (numFrames - i) * sizeof(Frame_t));
                return;
            }
            right = left;
            m_readPos += 2;
        } else {
            bytesRead = m_file.read((uint8_t*)&left, 2);
            if (bytesRead != 2) {
                m_valid = false;
                memset(&frames[i], 0, (numFrames - i) * sizeof(Frame_t));
                return;
            }
            bytesRead = m_file.read((uint8_t*)&right, 2);
            if (bytesRead != 2) {
                m_valid = false;
                memset(&frames[i], 0, (numFrames - i) * sizeof(Frame_t));
                return;
            }
            m_readPos += 4;
        }
        frames[i].left = left;
        frames[i].right = right;
    }
}
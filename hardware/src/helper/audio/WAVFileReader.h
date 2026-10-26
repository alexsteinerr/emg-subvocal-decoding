#ifndef WAVFILEREADER_H
#define WAVFILEREADER_H

#include "SampleSource.h"
#include <FS.h>

class WAVFileReader : public SampleSource {
public:
    WAVFileReader(fs::FS& fs, const char* path);
    ~WAVFileReader();

    int sampleRate() override { return m_sampleRate; }
    void getFrames(Frame_t* frames, int numFrames) override;
    bool isValid() const { return m_valid; }
    bool isFinished() override { return !m_valid; }

private:
    File m_file;
    int m_sampleRate;
    int m_numChannels;
    uint32_t m_dataStart;   // absolute position in file where PCM data begins
    uint32_t m_dataBytes;   // total number of PCM bytes
    uint32_t m_readPos;     // current offset from m_dataStart
    bool m_valid;
};

#endif
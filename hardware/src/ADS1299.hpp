#ifndef ADS1299_HPP
#define ADS1299_HPP

#include <Arduino.h>
#include <SPI.h>

// ADS1299 Register Map
#define R_ID        0x00
#define R_CONFIG1   0x01
#define R_CONFIG2   0x02
#define R_CONFIG3   0x03
#define R_CH1SET    0x05
#define R_CH2SET    0x06
#define R_CH3SET    0x07
#define R_CH4SET    0x08
#define R_CH5SET    0x09
#define R_CH6SET    0x0A
#define R_CH7SET    0x0B
#define R_CH8SET    0x0C
#define R_BIAS_SENSP 0x0D
#define R_BIAS_SENSN 0x0E
#define R_MISC1      0x15

// Commands
#define C_RESET     0x06
#define C_START     0x08
#define C_SDATAC    0x11
#define C_RDATAC    0x10
#define C_WREG      0x40

class ADS1299 {
private:
    int _pinCS, _pinDRDY, _pinSCLK, _pinMOSI, _pinMISO;



    SPIClass vspi;

    static constexpr uint32_t SPI_SPEED = 1000000;

public:
    long channelData[8];

    ADS1299(int cs, int drdy, int sclk, int mosi, int miso)
      : _pinCS(cs), _pinDRDY(drdy), _pinSCLK(sclk), _pinMOSI(mosi), _pinMISO(miso),
        vspi(FSPI) {}

    void begin() {
        pinMode(_pinCS, OUTPUT);
        digitalWrite(_pinCS, HIGH);
        pinMode(_pinDRDY, INPUT);

        vspi.begin(_pinSCLK, _pinMISO, _pinMOSI, _pinCS);

        // (Optional but recommended) Start/end transactions per operation,
        vspi.beginTransaction(SPISettings(SPI_SPEED, MSBFIRST, SPI_MODE1));

        setupRegisters();
    }

    bool isDataReady() {
        return (digitalRead(_pinDRDY) == LOW);
    }

    void updateData() {
        digitalWrite(_pinCS, LOW);
        delayMicroseconds(1);

        uint8_t rawData[27];
        for (int i = 0; i < 27; i++) {
            rawData[i] = vspi.transfer(0x00);
        }

        delayMicroseconds(1);
        digitalWrite(_pinCS, HIGH);

        for (int i = 0; i < 8; i++) {
            int byteIndex = 3 + (i * 3);
            long val = ((long)rawData[byteIndex] << 16) |
                       ((long)rawData[byteIndex + 1] << 8) |
                       (rawData[byteIndex + 2]);

            if (val & 0x800000) val |= 0xFF000000; // sign extend
            channelData[i] = val;
        }
    }

    void sendCommand(uint8_t cmd) {
        digitalWrite(_pinCS, LOW);
        delayMicroseconds(2);
        vspi.transfer(cmd);
        delayMicroseconds(2);
        digitalWrite(_pinCS, HIGH);
    }

    void writeRegister(uint8_t address, uint8_t value) {
        digitalWrite(_pinCS, LOW);
        delayMicroseconds(2);
        vspi.transfer(C_WREG | address);
        vspi.transfer(0x00);
        vspi.transfer(value);
        delayMicroseconds(2);
        digitalWrite(_pinCS, HIGH);
    }

private:
    void setupRegisters() {
        sendCommand(C_RESET);
        delay(100);

        sendCommand(C_SDATAC);
        delay(20);

        writeRegister(R_CONFIG1, 0x94); // 1000 Hz sample rate
        writeRegister(R_CONFIG2, 0xC0);
        writeRegister(R_CONFIG3, 0xFC);

        writeRegister(R_CH1SET, 0x60);
        writeRegister(R_CH2SET, 0x60);
        writeRegister(R_CH3SET, 0x60);
        writeRegister(R_CH4SET, 0x60);
        writeRegister(R_CH5SET, 0x60);

        writeRegister(R_CH6SET, 0x81);
        writeRegister(R_CH7SET, 0x81);
        writeRegister(R_CH8SET, 0x81);

        writeRegister(R_BIAS_SENSP, 0x1F);
        writeRegister(R_BIAS_SENSN, 0x1F);
        writeRegister(R_MISC1, 0x20);

        sendCommand(C_START);
        delay(10);
        sendCommand(C_RDATAC);
        delay(10);
    }
};

#endif

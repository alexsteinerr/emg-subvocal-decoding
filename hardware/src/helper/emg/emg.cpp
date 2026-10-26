#include "device.hpp"

#define ACTIVITY_THRESHOLD 50 

// Hardware Pins
#define PIN_CS    4    
#define PIN_SCLK  5    
#define PIN_MOSI  6    
#define PIN_MISO  7    
#define PIN_DRDY  15   

#define NUM_CHANNELS 5
#define BATCH_SIZE 50 

void Device::setup_ads(){
    ads = new ADS1299(PIN_CS, PIN_DRDY, PIN_SCLK, PIN_MOSI, PIN_MISO);
    ads->begin();
    
    Serial.println("[ADS1299] Initialized 5 Channels @ 1000Hz");
}

void Device::stream_eeg_data(){
  is_streaming = true;
  current_state = streaming_eeg;

  // 1. PREPARE HEADER
  string textMessage = "stream|" + string(AUTH_KEY)+ "|";
  size_t headerSize = textMessage.length();

  // 2. CALCULATE SIZES
  // 5 Channels * 50 Samples * 2 bytes = 500 bytes payload per packet
  size_t dataSize = NUM_CHANNELS * BATCH_SIZE * sizeof(int16_t);
  size_t totalSize = headerSize + dataSize;
  
  // 3. ALLOCATE BUFFERS ONCE 
  int16_t* batchBuffer = new int16_t[NUM_CHANNELS * BATCH_SIZE];
  uint8_t* packetBuffer = new uint8_t[totalSize];

  // Pre-fill header part of packetBuffer since it never changes
  memcpy(packetBuffer, textMessage.c_str(), headerSize);

  while(is_streaming){
    int sampleIndex = 0;
    long totalSignalEnergy = 0; 

    // COLLECTION LOOP (approx 50ms duration)
    while (sampleIndex < BATCH_SIZE) {
      if(ads->isDataReady()) {
          ads->updateData();
          
          // Cycle through 5 channels
          for(int ch = 0; ch < NUM_CHANNELS; ch++) {
              long sample24 = ads->channelData[ch];
              int16_t sample16 = (int16_t)(sample24 >> 8);
              
              // Interleave: [t0_ch0, t0_ch1... t0_ch4, t1_ch0...]
              int bufferPos = (sampleIndex * NUM_CHANNELS) + ch;
              batchBuffer[bufferPos] = sample16;

              totalSignalEnergy += abs(sample16);
          }
          sampleIndex++;
      }
      yield(); // Optional: Uncomment if ESP32 Watchdog triggers
    }

    // Average over 250 data points (50 samples * 5 chans)
    double averageActivity = totalSignalEnergy / (double)(BATCH_SIZE * NUM_CHANNELS);
    
    if (averageActivity < ACTIVITY_THRESHOLD) {
        Serial.println("[ADS1299] Signal flatline. Stopping stream.");
        is_streaming = false;
        current_state = waiting_for_trigger;
        break; 
    }

    // Copy the raw int16 data into the packet buffer after the header
    memcpy(packetBuffer + headerSize, batchBuffer, dataSize);

    // Send binary packet
    client.sendBIN(packetBuffer, totalSize);
  }

  // Cleanup
  delete[] batchBuffer;
  delete[] packetBuffer;
}

// Helper: Get snippet for pattern recognition
// 1000 samples @ 1000Hz = 1.0 second snapshot
int16_t* Device::get_eeg_sample() {
  int samplesToRead = 1000; 
  int16_t* buffer = (int16_t*)malloc(samplesToRead * NUM_CHANNELS * sizeof(int16_t));
  int count = 0;

  while (count < samplesToRead) {
    if(ads->isDataReady()) {
        ads->updateData();
        for(int ch = 0; ch < NUM_CHANNELS; ch++) {
            long sample24 = ads->channelData[ch];
            buffer[(count * NUM_CHANNELS) + ch] = (int16_t)(sample24 >> 8);
        }
        count++;
    }
  }
  return buffer;
}
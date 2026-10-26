#include "device.hpp"
#include "enr_client.h" 
#include <Arduino.h>
#include "ADS1299.hpp"
#include <LittleFS.h>
#include <Wire.h>
#include <vector>
#include <cmath>

Device device;
ENRClient enr;

void Device::wait_for_eeg_trigger(){
    /*
    device.current_state = device.waiting_for_trigger;

    Serial.println("[EMG] Waiting for trigger pattern...");
    while(!device.predict(device.get_eeg_sample())){
        device.client.loop();
        if(device.is_streaming) device.stream_camera_to_ws();
    }
    Serial.println("[EMG] Trigger detected! Recording session...");
    device.stream_eeg_data();
    */
}

void test(){
    string textMessage = "stream|" + string(device.AUTH_KEY)+ "|";
    size_t headerSize = textMessage.length();

    size_t dataSize = 50 * sizeof(int16_t);
    size_t totalSize = headerSize + dataSize;
    
    int16_t* batchBuffer = new int16_t[50];
    uint8_t* packetBuffer = new uint8_t[totalSize];

    memcpy(packetBuffer, textMessage.c_str(), headerSize);

    int sampleIndex = 0;
    long totalSignalEnergy = 0;

    // Copy the raw int16 data into the packet buffer after the header
    memcpy(packetBuffer + headerSize, batchBuffer, dataSize);

    // Send binary packet
    device.client.sendBIN(packetBuffer, totalSize);

    delete[] batchBuffer;
    delete[] packetBuffer;
}

void webSocketEvent(WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            if (device.current_state != device.not_connected) { 
                Serial.println("[WSc] Disconnected!");
                device.current_state = device.not_connected;
            }
            break;

        case WStype_CONNECTED:
            Serial.println("[WSc] Connected to url: /ws");
            device.current_state = device.connected;
            device.client.sendTXT(("authenticate|" + device.AUTH_KEY).c_str());
            break;

        case WStype_TEXT:
        {
            String message = String((char*)payload).substring(0, length);
            vector<string> parts = device.split(message.c_str(), "|");

            if(parts.size() <= 1) break;;

            if(parts[0] == "volume" && parts.size() == 2){
                device.set_volume(parts[1]);
            }
            break;
        }

        case WStype_BIN:
        {
            if(length == 0) break;

            // TTS audio stream framing:
            //   0x01           - stream start (informational)
            //   0x02           - stream end   (informational)
            //   0x03 + PCM...  - raw PCM chunk: 22050 Hz, mono, 16-bit,
            if (payload[0] == 0x01) {
                Serial.println("[TTS] Stream started");
                device.audioPlayer.resetStream(TARGET_SPEAKER); 
                break;
            }

            if (payload[0] == 0x02) {
                Serial.println("[TTS] Stream ended");
                device.audioPlayer.endStream(TARGET_SPEAKER);  
                break;
            }

            if (payload[0] == 0x03) {
                //device.audioPlayer.playBuffer(payload + 1, length - 1, TARGET_SPEAKER);
                break;
            }

            break;
        }

        case WStype_ERROR:
            Serial.println("[WSc] WebSocket error!");
            break;
    }
}

void send_encrypted_data(String raw_input) {
    String encrypted_json = enr.prepare_payload(raw_input);
    String msg = "send_data¬" + String(device.AUTH_KEY.c_str()) + "¬" + encrypted_json;
    device.client.sendTXT(msg);
}

void setup() {
    Serial.begin(115200);

    unsigned long t0 = millis();
    while (!Serial && millis()-t0 < 5000)
        delay(10);

    Serial.println();

    if (!psramInit())
        Serial.println("[ERROR] psramInit() failed");

    if (psramFound())
        Serial.printf("[INFO] PSRAM OK, size: %d bytes\n", ESP.getPsramSize());
    else {
        Serial.println("[ERROR] PSRAM NOT FOUND");
        while(1) delay(1000);
    }

    Serial.println("[INFO] Device booted");

    if(!LittleFS.begin(true)){
        Serial.println("[ERROR] LittleFS Mount Failed");
        return;
    }

    device.preferences.begin("device", false); 

    // Set from preferences loaded on build
    device.save_string("session_key", "a1b2c3d4e5f60718293a4b5c6d7e8f901a2b3c4d5e6f0718293a4b5c6d7e8f90");
    device.AUTH_KEY = device.read_string("auth_key").c_str();

    // AUDIO – must be first
    device.audioPlayer.init();
    device.set_volume("100");           
    device.audioPlayer.playFile("/boot.wav", TARGET_SPEAKER);

    String session_key_hex = device.read_string("session_key"); 

    if(session_key_hex != "error" && session_key_hex.length() == 64) {
        enr.begin(session_key_hex);
        Serial.println("[ENR] Encryption initialized.");
    }
    else {
        Serial.println("[ENR] Warning: session_key not found.");
    }

    if(device.AUTH_KEY == "error"){
        Serial.println("[ERROR] Device not initialized!");
        return;
    }

    Serial.println("[INFO] Device initialized");

    device.setup_ble();

    if(device.read_string("ssid") != "error" && device.read_string("password") != "error")
        device.connect_wifi(device.read_string("ssid").c_str(), device.read_string("password").c_str());
    else device.invoke_error("WiFi not connected");

    Serial.println("[WSc] Connecting to secure WebSocket...");

    device.client.beginSSL(
        "fromintent.org",
        443,
        "/"
    );

    device.client.onEvent(webSocketEvent);
    device.client.setReconnectInterval(1000);

    Serial.println("[WSc] Waiting for TLS/WebSocket connection...");

    unsigned long ws_start = millis();

    while(device.current_state != device.connected && millis() - ws_start < 15000) {
        device.client.loop();
        delay(10);
    }

    if(device.current_state != device.connected) { Serial.println("[WSc] Connection failed!"); } 
    else { Serial.println("[WSc] TLS/WebSocket connected!"); }

    //device.setup_tf();
    //device.setup_ads();

    Serial.println("[INFO] Device setup complete!");

    //device.check_battery();
}

void loop() {
    // OTA
    if (device.current_state == device.ota) {
        return;
    }

    // ALWAYS service WebSocket
    device.client.loop();
}
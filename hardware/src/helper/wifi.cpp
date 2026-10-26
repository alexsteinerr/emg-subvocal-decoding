#include "device.hpp"

/**
 * Connect device to WiFi, only WPA2-Enterprise and WPA3-Enterprise are supported
 * 
 * @param ssid ssid
 * @param password password
*/
void Device::connect_wifi(const char* ssid, const char* password) {
    // Ensure clean state without double deinit
    WiFi.disconnect(true);  
    delay(100);
    WiFi.mode(WIFI_STA);
    WiFi.setAutoConnect(true);
    
    audioPlayer.playFile("/connecting.wav", TARGET_SPEAKER);
    
    WiFi.begin(ssid, password);
    Serial.print("[WIFI] Connecting");
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 40) {
        delay(500);
        Serial.print(".");
        attempts++;

        if (attempts % 6 == 0) {   // every 3 seconds
            Serial.println();
            audioPlayer.playFile("/loading.wav", TARGET_SPEAKER);
        }
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\n[WiFi] Connected to WiFi");
        Serial.print("[WIFI] IP: ");
        Serial.println(WiFi.localIP());
        audioPlayer.playFile("/success.wav", TARGET_SPEAKER);  
    } else {
        Serial.println("\n[ERROR] WiFi connection failed!");
        audioPlayer.playFile("/connection_failed.wav", TARGET_SPEAKER);    
        invoke_error("WiFi connection failed");
    }
}

void Device::disconnect_wifi(){
    WiFi.disconnect(true); 
}
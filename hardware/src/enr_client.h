#ifndef ENR_CLIENT_H
#define ENR_CLIENT_H

#include <Arduino.h>
#include "mbedtls/gcm.h"
#include "mbedtls/base64.h"
#include "esp_system.h" // for esp_random

class ENRClient {
private:
    unsigned char session_key[32]; // AES-256 requires 32 bytes
    bool key_initialized = false;

    // OPTIMIZED: Uses const reference and avoids String::substring to prevent Heap crashes
    void hexStringToBytes(const String& hex, unsigned char* out) {
        for (unsigned int i = 0; i < hex.length(); i += 2) {
            // Create a temporary standard C buffer for 2 chars + null terminator
            char buf[3]; 
            buf[0] = hex[i];
            buf[1] = hex[i+1]; // Safe because we checked length is 64 (even)
            buf[2] = '\0';
            
            // Convert simple C-string to byte
            out[i / 2] = (unsigned char)strtol(buf, NULL, 16);
        }
    }

    // Helper for Base64 Encoding
    String base64Encode(const unsigned char* input, size_t inputLen) {
        size_t outputLen = 0;
        mbedtls_base64_encode(NULL, 0, &outputLen, input, inputLen); // Get size
        
        unsigned char* buffer = new unsigned char[outputLen + 1];
        mbedtls_base64_encode(buffer, outputLen + 1, &outputLen, input, inputLen);
        buffer[outputLen] = '\0'; // Null terminate
        
        String result = (char*)buffer;
        delete[] buffer;
        return result;
    }

public:
    ENRClient() {}

    // OPTIMIZED: Pass by const reference to avoid unnecessary copying
    void begin(const String& hexKey) {
        if (hexKey.length() != 64) {
            Serial.println("[ENR] Error: Session key must be 64 hex characters (32 bytes)");
            return;
        }
        hexStringToBytes(hexKey, session_key);
        key_initialized = true;
    }

    // Encrypt data and return JSON string
    String prepare_payload(String plaintext) {
        if (!key_initialized) return "{}";

        mbedtls_gcm_context gcm;
        mbedtls_gcm_init(&gcm);

        int ret = mbedtls_gcm_setkey(&gcm, MBEDTLS_CIPHER_ID_AES, session_key, 256);
        if (ret != 0) {
            Serial.println("[ENR] GCM setkey failed");
            return "{}";
        }

        // 1. Generate Nonce (12 bytes)
        unsigned char nonce[12];
        for (int i = 0; i < 12; i++) {
            nonce[i] = (unsigned char)(esp_random() & 0xFF);
        }

        // 2. Prepare buffers
        size_t plainLen = plaintext.length();
        size_t tagLen = 16; 
        unsigned char* outputBuffer = new unsigned char[plainLen + tagLen];
        unsigned char* tagBuffer = outputBuffer + plainLen; 

        // 3. Encrypt
        ret = mbedtls_gcm_crypt_and_tag(&gcm, MBEDTLS_GCM_ENCRYPT, plainLen, 
                                        nonce, 12, 
                                        NULL, 0, 
                                        (const unsigned char*)plaintext.c_str(), 
                                        outputBuffer, 
                                        tagLen, tagBuffer);

        mbedtls_gcm_free(&gcm);

        if (ret != 0) {
            Serial.printf("[ENR] Encryption failed: -0x%04x\n", -ret);
            delete[] outputBuffer;
            return "{}";
        }

        // 4. Base64 Encode components
        String nonceB64 = base64Encode(nonce, 12);
        String ciphertextB64 = base64Encode(outputBuffer, plainLen + tagLen);

        delete[] outputBuffer;

        // 5. Construct JSON
        String json = "{";
        json += "\"nonce\":\"" + nonceB64 + "\",";
        json += "\"ciphertext\":\"" + ciphertextB64 + "\"";
        json += "}";

        return json;
    }
};

#endif
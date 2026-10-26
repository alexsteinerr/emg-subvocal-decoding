#include "../device.hpp"

static const char* rootCACertificate = \
"-----BEGIN CERTIFICATE-----\n" \
"MIICCTCCAY6gAwIBAgINAgPlwGjvYxqccpBQUjAKBggqhkjOPQQDAzBHMQswCQYD\n" \
"VQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZpY2VzIExMQzEUMBIG\n" \
"A1UEAxMLR1RTIFJvb3QgUjQwHhcNMTYwNjIyMDAwMDAwWhcNMzYwNjIyMDAwMDAw\n" \
"WjBHMQswCQYDVQQGEwJVUzEiMCAGA1UEChMZR29vZ2xlIFRydXN0IFNlcnZpY2Vz\n" \
"IExMQzEUMBIGA1UEAxMLR1RTIFJvb3QgUjQwdjAQBgcqhkjOPQIBBgUrgQQAIgNi\n" \
"AATzdHOnaItgrkO4NcWBMHtLSZ37wWHO5t5GvWvVYRg1rkDdc/eJkTBa6zzuhXyi\n" \
"QHY7qca4R9gq55KRanPpsXI5nymfopjTX15YhmUPoYRlBtHci8nHc8iMai/lxKvR\n" \
"HYqjQjBAMA4GA1UdDwEB/wQEAwIBhjAPBgNVHRMBAf8EBTADAQH/MB0GA1UdDgQW\n" \
"BBSATNbrdP9JNqPV2Py1PsVq8JQdjDAKBggqhkjOPQQDAwNpADBmAjEA6ED/g94D\n" \
"9J+uHXqnLrmvT/aDHQ4thQEd0dlq7A/Cr8deVl5c1RxYIigL9zC2L7F8AjEA8GE8\n" \
"p/SgguMh1YQdc4acLa/KNJvxn7kjNuK8YAOdgLOaVsjh4rsUecrNIdSUtUlD\n" \
"-----END CERTIFICATE-----\n";

void Device::check_for_update() {
    String currentVer = String(get_version().c_str());
    String body = "{\"AUTH_KEY\":\"" + String(AUTH_KEY.c_str()) + "\"}";

    WiFiClientSecure versionClient;
    versionClient.setCACert(rootCACertificate);
    versionClient.setTimeout(30000);

    HTTPClient http;
    http.begin(versionClient, "https://ota.fromintent.org/firmware/version");
    http.addHeader("Content-Type", "application/json");

    int code = http.POST(body);

    if (code != HTTP_CODE_OK) {
        Serial.printf("[OTA] Version check failed, HTTP code: %d\n", code);
        http.end();
        ESP.restart();
        return;
    }

    String latestVersion = http.getString();
    latestVersion.trim();

    Serial.println("[OTA] Current: " + currentVer);
    Serial.println("[OTA] Latest:  " + latestVersion);

    http.end();

    if (latestVersion == currentVer) {
        Serial.println("[OTA] Firmware is up to date.");
        return;
    }

    Serial.println("[OTA] New firmware available. Downloading...");

    if (current_state == connected) {
        client.disconnect();
        client = WebSocketsClient();
        current_state = not_connected;
        close_ble();
        delay(1000);
    }

    WiFiClientSecure downloadClient;
    downloadClient.setCACert(rootCACertificate);
    downloadClient.setTimeout(30000);

    HTTPClient downloadHttp;
    downloadHttp.begin(downloadClient, "https://ota.fromintent.org/firmware/latest.bin");
    downloadHttp.addHeader("Content-Type", "application/json");

    Serial.println("[OTA] Requesting firmware...");

    int downloadCode = downloadHttp.POST(body);

    if (downloadCode == HTTP_CODE_OK) {
        int contentLength = downloadHttp.getSize();
        Serial.printf("[OTA] Firmware size: %d bytes\n", contentLength);

        if (contentLength > 0 && Update.begin(contentLength)) {
            size_t written = Update.writeStream(downloadHttp.getStream());

            if (written == contentLength) {
                Serial.println("[OTA] Download complete. Finalizing...");

                if (Update.end()) {
                    if (Update.isFinished()) {
                        Serial.println("[OTA] Update successful! Rebooting...");
                        downloadHttp.end();
                        delay(500);
                        ESP.restart();
                    } else {
                        Serial.println("[OTA] Update not finished!");
                    }
                } else {
                    Serial.printf("[OTA] Update finalization failed. Error: %u\n", Update.getError());
                }
            } else {
                Serial.printf("[OTA] Write mismatch: wrote %d of %d bytes\n", written, contentLength);
                Update.abort();
            }
        } else {
            Serial.println("[OTA] Update begin failed (check flash space)");
        }
    } else {
        Serial.printf("[OTA] Download failed, HTTP code: %d\n", downloadCode);
    }

    downloadHttp.end();
}

std::string Device::get_version() {
    return version;
}
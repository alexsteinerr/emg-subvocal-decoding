#include "device.hpp"

#define SERVICE_UUID "0cd36874-33d0-4056-9d64-a3f8a5749d64"

bool deviceConnected = false;
BLECharacteristic *pCharacteristic;

class BleCallbacks : public BLEServerCallbacks {
private:
  Device* device;

public:
  BleCallbacks(Device* d) : device(d) {}

  void onConnect(BLEServer* pServer) override {
    deviceConnected = true;
    Serial.println("[BLE] Device connected");
  }

  void onDisconnect(BLEServer* pServer) override {
    deviceConnected = false;
    Serial.println("[BLE] Device disconnected");
    pServer->startAdvertising();
  }
};

/**
 * Setup BleDevice and Server
 * Create Characteristic & Callbacks
 */
void Device::setup_ble() {
  BLEDevice::init("Intent");

  BLEServer *pServer = BLEDevice::createServer();
  pServer->setCallbacks(new BleCallbacks(this));

  String charUUID = read_string("CHAR_UUID");

  if (charUUID == "error") {
    charUUID = generate_uuid();
    save_string("CHAR_UUID", charUUID.c_str());
  }

  Serial.println("[BLE] Characteristic UUID: " + charUUID);

  BLEService *pService =
    pServer->createService(SERVICE_UUID);

  pCharacteristic =
    pService->createCharacteristic(
      charUUID.c_str(),
      BLECharacteristic::PROPERTY_READ |
      BLECharacteristic::PROPERTY_WRITE |
      BLECharacteristic::PROPERTY_NOTIFY
    );

  pCharacteristic->setValue("Hello World");
  pCharacteristic->setCallbacks(this);
  pCharacteristic->addDescriptor(new BLE2902());

  pService->start();

  BLEAdvertising *pAdvertising = BLEDevice::getAdvertising();

  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);
  pAdvertising->setMinPreferred(0x12);

  BLEDevice::startAdvertising();

  Serial.println("[BLE] Advertising started");
}

/**
 * Process received Ble Data
 * 
 * @param data string received
 */
void Device::process_ble_data(std::string data){
  vector<string> message_parts = split(data.c_str(), "|");
  if (message_parts[0] == "authentication_key" && message_parts.size() == 2) {
    if (AUTH_KEY.length() > 0) {
      return;
    }

    AUTH_KEY = message_parts[1].c_str();
    save_string("auth_key", message_parts[1]);
    Serial.println("[INFO] Device authenticated");

    if (WiFi.status() == WL_CONNECTED) {
      std::string ble_data = "ip|" + AUTH_KEY + "|" + WiFi.localIP().toString().c_str();
      send_ble(ble_data);  
    }
  } else if (message_parts[0] == "wifi" && message_parts[1] == AUTH_KEY.c_str() && message_parts.size() == 4) {
    save_string("ssid", message_parts[2]);
    save_string("password", message_parts[3]);

    disconnect_wifi();
    connect_wifi(message_parts[2].c_str(), message_parts[3].c_str());

    std::string ble_data = "ip|" + AUTH_KEY + "|" + WiFi.localIP().toString().c_str();
    send_ble(ble_data);  
  } else if (message_parts[0] == "firmware" && message_parts[1] == AUTH_KEY.c_str() && message_parts.size() == 2){
    std::string firmware_data = "firmware|AUTH_KEY|1.0.0";
    send_ble(firmware_data);
  } else if (message_parts[0] == "unpair" && message_parts[1] == AUTH_KEY.c_str() && message_parts.size() == 2){
    Serial.println("[INFO] Device unpaired");
    preferences.clear();
    disconnect_wifi();
    AUTH_KEY = "";
  } else { // if the message isn't any of the above the ble received is mirrored to the dart ws 
    static uint8_t mirror_buffer[512];   
    if (data.size() < sizeof(mirror_buffer)) {
      memcpy(mirror_buffer, data.data(), data.size());
      client.sendBIN(mirror_buffer, data.size());
    } else {
      Serial.println("[process_ble_data] Data too large to mirror");
    }
  }
}

/**
 * On message received by the BLE clienet connected
 */
void Device::onWrite(BLECharacteristic *pCharacteristic) {
  std::string data = pCharacteristic->getValue();
        
  if (data.length() > 0) {
    process_ble_data(data);
  }
}

/**
 * Send message to Bluethoot
 * 
 * @param payload message to send
 */
void Device::send_ble(const std::string& payload) {
  static uint8_t ble_buffer[512];

  if (deviceConnected && pCharacteristic != nullptr) {
    pCharacteristic->setValue((uint8_t*)payload.data(), payload.size());
    pCharacteristic->notify();
  } else {
    if (!client.isConnected()) {
      Serial.println("[BLE] No BLE and WebSocket not connected");
      return;
    }

    std::string error_msg = "ble_error|" + AUTH_KEY + "|";
    if (error_msg.size() >= sizeof(ble_buffer)) {
      Serial.println("[BLE] Error message too large");
      return;
    }

    memcpy(ble_buffer, error_msg.data(), error_msg.size());
    client.sendBIN(ble_buffer, error_msg.size());  
  }
}

void Device::close_ble() {
  BLEDevice::stopAdvertising();
  deviceConnected = false;

  BLEDevice::deinit();
  pCharacteristic = nullptr;

  Serial.println("[BLE] Closed and deinitialized");
}
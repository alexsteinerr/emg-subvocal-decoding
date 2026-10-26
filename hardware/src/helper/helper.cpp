#include "device.hpp"

void Device::save_string(String key, const string value){
  preferences.putString(key.c_str(), value.c_str());
}

String Device::read_string(String key){
  return preferences.getString(key.c_str(), "error");
}

String Device::generate_uuid() {
  uint32_t a = esp_random();
  uint32_t b = esp_random();
  uint32_t c = esp_random();
  uint32_t d = esp_random();

  char uuid[37];

  snprintf(
    uuid,
    sizeof(uuid),
    "%08lX-%04lX-%04lX-%04lX-%04lX%08lX",
    (unsigned long)a,
    (unsigned long)(b & 0xFFFF),
    (unsigned long)((c & 0x0FFF) | 0x4000),
    (unsigned long)((d & 0x3FFF) | 0x8000),
    (unsigned long)(d & 0xFFFF),
    (unsigned long)c
  );

  return String(uuid);
}
vector<string> Device::split(string s, string delimiter){
    vector<string> list;
    size_t pos = 0;
    string token;
    while ((pos = s.find(delimiter)) != string::npos) {
        token = s.substr(0, pos);
        list.push_back(token);
        s.erase(0, pos + delimiter.length());
    }
    list.push_back(s);
    return list;
}

void Device::check_battery(){
  uint32_t Vbatt = 0;

  for (int i = 0; i < 16; i++) {
    Vbatt += analogReadMilliVolts(A0);  
  }

  int battery_percentage = constrain(((2 * Vbatt / 16000.0 - minBatteryVoltage) / (maxBatteryVoltage - minBatteryVoltage)) * 100, 0, 100);

  //send_ble((char*)("battery|" + std::string(AUTH_KEY) + "|" + std::to_string(battery_percentage)).c_str());
}
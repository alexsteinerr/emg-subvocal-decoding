#include "device.hpp"

void Device::invoke_error(const char* err){
   Serial.println(err);
}
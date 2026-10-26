#include "device.hpp"
#include "driver/i2s.h"
#include "AudioConfig.h"

void Device::set_volume(std::string volume_input) {
    int vol = std::stoi(volume_input);
    vol = constrain(vol, 0, 100);
    volume = vol;
    audioPlayer.setVolume(vol);

    std::string message = "volume|" + std::string(AUTH_KEY) + "|" + std::to_string(volume);
    send_ble(const_cast<char*>(message.c_str()));
}
#pragma once

#include <chrono>
#include <string>

namespace megamol::core::utility {
std::string serialize_timestamp(std::chrono::system_clock::time_point const& tp);
}

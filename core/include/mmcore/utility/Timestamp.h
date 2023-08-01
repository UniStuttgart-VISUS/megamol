/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <chrono>
#include <string>

namespace megamol::core::utility {
std::string serialize_timestamp(std::chrono::system_clock::time_point const& tp);
}

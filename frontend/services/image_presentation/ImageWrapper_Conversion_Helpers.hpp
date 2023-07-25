/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <vector>

namespace megamol::frontend_resources {

using byte = unsigned char;

namespace conversion {

unsigned int to_uint(void* ptr);

std::vector<byte>* to_vector(void* ptr);
} // namespace conversion

} // namespace megamol::frontend_resources

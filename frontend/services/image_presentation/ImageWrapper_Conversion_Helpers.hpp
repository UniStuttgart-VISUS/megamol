/*
 * ImageWrapper_Conversion_Helpers.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
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

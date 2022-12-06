/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <cstdint>
#include <vector>

#include "Framebuffer.h"

namespace megamol::core::view {

struct CPUFramebufferData {
    unsigned int col_tex = 0;
    unsigned int depth_tex = 0;
};

using CPUFramebuffer = Framebuffer<std::vector<uint32_t>, std::vector<float>, CPUFramebufferData>;

} // namespace megamol::core::view

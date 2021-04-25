/*
 * CPUFramebuffer.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include <vector>

#include "Framebuffer.h"

namespace megamol::core::view {

using CPUFramebuffer = Framebuffer<std::vector<uint32_t>, std::vector<float>>;

}

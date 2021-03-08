/*
 * CPUFramebuffer.h
 *
 * Copyright (C) 2021 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once
#include <vector>

namespace megamol::core::view {

struct CPUFramebuffer {
    bool depthBufferActive = false;
    std::vector<uint32_t> colorBuffer;
    std::vector<float> depthBuffer;
    unsigned int width = 0;
    unsigned int height = 0;
    int x = 0;
    int y = 0;
};

}

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

namespace megamol::core::view {

template<typename CB, typename DB, typename UD = void*>
struct Framebuffer {
    bool depthBufferActive = false;
    CB colorBuffer;
    DB depthBuffer;
    unsigned int width = 0;
    unsigned int height = 0;
    int x = 0;
    int y = 0;
    UD data;

    unsigned int getWidth() {
        return width;
    }

    unsigned int getHeight() {
        return height;
    }
};

} // namespace megamol::core::view

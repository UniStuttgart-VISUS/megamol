/*
 * WindowManipulation.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

namespace megamol {
namespace frontend_resources {

struct WindowManipulation {
    void set_window_title(const char* title) const;
    void set_framebuffer_size(const unsigned int width, const unsigned int height) const;
    void set_window_position(const unsigned int width, const unsigned int height) const;

    enum class Fullscreen {
        Maximize,
        Restore
    };
    void set_fullscreen(const Fullscreen action) const;

    void* window_ptr = nullptr;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */

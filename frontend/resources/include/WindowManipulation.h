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
    void set_swap_interval(const unsigned int wait_frames) const; // DANGER: assumes there is a GL context active
    std::function<void(const int)> set_mouse_cursor;

    enum class Fullscreen {
        Maximize,
        Restore
    };
    void set_fullscreen(const Fullscreen action) const;

    void* window_ptr = nullptr;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */

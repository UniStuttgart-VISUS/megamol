/*
 * WindowManipulation.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#define NON_GL_EMPTY ;
#ifndef WITH_GL
#define NON_GL_EMPTY {}
#endif


namespace megamol {
namespace frontend_resources {

struct WindowManipulation {
    void set_window_title(const char* title) const NON_GL_EMPTY
    void set_framebuffer_size(const unsigned int width, const unsigned int height) const NON_GL_EMPTY
    void set_window_position(const unsigned int width, const unsigned int height) const NON_GL_EMPTY
    void set_swap_interval(const unsigned int wait_frames) const NON_GL_EMPTY // DANGER: assumes there is a GL context active
    std::function<void(const int)> set_mouse_cursor;

    void swap_buffers() const NON_GL_EMPTY

    enum class Fullscreen {
        Maximize,
        Restore
    };
    void set_fullscreen(const Fullscreen action) const NON_GL_EMPTY

    void* window_ptr = nullptr;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */

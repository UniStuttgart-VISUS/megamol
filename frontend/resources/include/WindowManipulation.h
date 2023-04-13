/*
 * WindowManipulation.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "GL_STUB.h"

namespace megamol::frontend_resources {

static std::string WindowManipulation_Req_Name = "WindowManipulation";

struct WindowManipulation {
    void set_window_title(const char* title) const GL_STUB();
    void set_framebuffer_size(const unsigned int width, const unsigned int height) const GL_STUB();
    void set_window_position(const unsigned int width, const unsigned int height) const GL_STUB();
    void set_swap_interval(const unsigned int wait_frames) const
        GL_STUB(); // DANGER: assumes there is a GL context active
    std::function<void(const int)> set_mouse_cursor;

    void swap_buffers() const GL_STUB();

    enum class Fullscreen { Maximize, Restore };
    void set_fullscreen(const Fullscreen action) const GL_STUB();

    void* window_ptr = nullptr;
};

} // namespace megamol::frontend_resources

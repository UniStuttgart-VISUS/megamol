/*
 * Framebuffer_Events.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <vector>

namespace megamol {
namespace frontend_resources {

struct FramebufferState {
    int width = 0;
    int height = 0;

    bool operator==(const FramebufferState& other) {
        return this->width == other.width && this->height == other.height;
    }
    bool operator!=(const FramebufferState& other) { return !(*this == other); }
};
struct FramebufferEvents {
    std::vector<FramebufferState> size_events;

    FramebufferState previous_state;

    bool is_resized() { return size_events.size() && previous_state != size_events.back(); }

    void apply_state() {
        if (size_events.size()) {
            this->previous_state.width = size_events.back().width;
            this->previous_state.height = size_events.back().height;
        }
    }
    void clear() {
        apply_state();
        size_events.clear();
    }
};

namespace input = frontend_resources;

} /* end namespace frontend_resources */
} /* end namespace megamol */

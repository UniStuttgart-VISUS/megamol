/**
 * MegaMol
 * Copyright (c) 2020, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <vector>

namespace megamol::frontend_resources {

struct FramebufferState {
    int width = 1;
    int height = 1;

    bool operator==(const FramebufferState& other) const {
        return this->width == other.width && this->height == other.height;
    }
    bool operator!=(const FramebufferState& other) const {
        return !(*this == other);
    }
};
struct FramebufferEvents {
    std::vector<FramebufferState> size_events;

    FramebufferState previous_state;

    bool is_resized() {
        return !size_events.empty() && previous_state != size_events.back();
    }

    void apply_state() {
        if (!size_events.empty()) {
            this->previous_state.width = size_events.back().width;
            this->previous_state.height = size_events.back().height;
        }
    }
    void clear() {
        apply_state();
        size_events.clear();
    }

    void append(FramebufferEvents const& other) {
        size_events.insert(size_events.end(), other.size_events.begin(), other.size_events.end());
    }
};

namespace input = frontend_resources;

} // namespace megamol::frontend_resources

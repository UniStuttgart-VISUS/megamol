#pragma once

#include "mmcore/view/Input.h"

#include <imgui.h>

namespace megamol::gui {
struct BasicConfig {
    bool show = false;                          // [SAVED] show/hide window
    ImGuiWindowFlags flags = 0;                 // [SAVED] imgui window flags
    megamol::core::view::KeyCode hotkey;        // [SAVED] hotkey for opening/closing window
    ImVec2 position = ImVec2(0.0f, 0.0f);       // [SAVED] position for reset on state loading (current position)
    ImVec2 size = ImVec2(0.0f, 0.0f);           // [SAVED] size for reset on state loading (current size)
    ImVec2 reset_size = ImVec2(0.0f, 0.0f);     // [SAVED] minimum window size for soft reset
    ImVec2 reset_position = ImVec2(0.0f, 0.0f); // [SAVED] window position for minimize reset
    bool collapsed = false;                     // [SAVED] flag indicating whether window is collapsed or not
    bool reset_pos_size = true;                 // flag indicates whether to reset window position and size
};
} // namespace megamol::gui

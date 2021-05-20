/*
 * GUIRegisterWindow.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "WindowCollection.h"

#define VALIDATE_IMGUI_SCOPE { if (!((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false))) return; }


namespace megamol {
namespace frontend_resources {

    struct GUIRegisterWindow {

        // Register GUI window rendering callback
        /// ! Make sure to call only once to prevent performance overhead !
        std::function<void(const std::string&, std::function<void(megamol::gui::WindowConfiguration::Basic&)> )> register_window;
    };

} /* end namespace frontend_resources */
} /* end namespace megamol */

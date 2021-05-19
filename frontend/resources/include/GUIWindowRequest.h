/*
 * GUIWindowRequest.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "WindowCollection.h"

#define VALID_IMGUI_SCOPE { if (!((ImGui::GetCurrentContext() != nullptr) ? (ImGui::GetCurrentContext()->WithinFrameScope) : (false))) return; }


namespace megamol {
namespace frontend_resources {

    struct GUIWindowRequest {

        // Register GUI window rendering callback
        std::function<void(const std::string&, std::function<void(megamol::gui::WindowConfiguration::Basic&)>& )> register_window;
    };

} /* end namespace frontend_resources */
} /* end namespace megamol */

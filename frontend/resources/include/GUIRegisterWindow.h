/*
 * GUIRegisterWindow.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "windows/WindowCollection.h"

namespace megamol::frontend_resources {

static std::string GUIRegisterWindow_Req_Name = "GUIRegisterWindow";

struct GUIRegisterWindow {

    /// ! Make sure to call the following register functions only once to prevent performance overhead,
    /// e.g. call in setRequestedResources() of frontend service.

    // Register GUI window rendering callback
    // Parameters: window name, rendering callback function
    std::function<void(const std::string&, std::function<void(megamol::gui::AbstractWindow::BasicConfig&)>)>
        register_window;

    // Register GUI pop-up rendering callback
    // Parameters: Pop-up name, open opop-up flag, rendering callback function
    std::function<void(const std::string&, std::weak_ptr<bool>, std::function<void(void)>)> register_popup;

    // Register GUI notification popup
    // Parameters: Pop-up name, open pop-up flag, message
    std::function<void(const std::string&, std::weak_ptr<bool>, const std::string&)> register_notification;
};

} // namespace megamol::frontend_resources

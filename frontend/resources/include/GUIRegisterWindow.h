/*
 * GUIRegisterWindow.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "WindowCollection.h"


namespace megamol {
namespace frontend_resources {

    struct GUIRegisterWindow {

        /// ! Make sure to call the following register functions only once to prevent performance overhead,
        /// e.g. call in setRequestedResources() of frontend service.

        // Register GUI window rendering callback
        // Parameters: window name, rendering callback function
        std::function<void(const std::string&, std::function<void(megamol::gui::WindowConfiguration::Basic&)> )> register_window;

        // Register GUI pop-up rendering callback
        // Parameters: Pop-up name, open opop-up flag, rendering callback function
        std::function<void(const std::string&, bool&, std::function<void(void)> )> register_popup;

        // Register GUI notification popup
        // Parameters: Pop-up name, message
        std::function<void(const std::string&, bool&, const std::string&)> register_notification;
    };

} /* end namespace frontend_resources */
} /* end namespace megamol */

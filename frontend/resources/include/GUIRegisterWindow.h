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

        // Register GUI window rendering callback
        /// ! Make sure to call only once to prevent performance overhead, e.g. call in setRequestedResources() of frontend service.
        std::function<void(const std::string&, std::function<void(megamol::gui::WindowConfiguration::Basic&)> )> register_window;

        // Register GUI popup rendering callback
        /// ! Make sure to call only once to prevent performance overhead, e.g. call in setRequestedResources() of frontend service.
        std::function<void(const std::string&, bool&, std::function<void(void)> )> register_popup;
    };

} /* end namespace frontend_resources */
} /* end namespace megamol */

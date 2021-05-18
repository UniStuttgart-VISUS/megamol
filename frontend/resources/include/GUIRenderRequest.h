/*
 * GUIRenderRequest.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "WindowCollection.h"
typedef megamol::gui::WindowCollection::WindowConfiguration WinConfig_t;

namespace megamol {
namespace frontend_resources {

    struct GUIRenderRequest {

        // Provide unique window name
        std::string window_name;

        // Register GUI window rendering callback
        std::function<void(WinConfig_t&)> callback = [&](WinConfig_t&){ };
    };

} /* end namespace frontend_resources */
} /* end namespace megamol */

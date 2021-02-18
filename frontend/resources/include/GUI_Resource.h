/*
 * GUI_Resource.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

namespace megamol {
namespace frontend_resources {

struct GUIResource {

    // Request updated GUI state (e.g. ScreenshotService saves project with GUI state tp PNG header)
    std::function<std::string(void)> request_gui_state = [&](void){ return std::string(); };

    // Provide GUI state as JSON string (e.g. Lua_Service_Wrapper loads project providing GUI state via mmSetGUIState)
    std::function<void(std::string)> provide_gui_state = [&](std::string) { };

    // Provide GUI visibility (e.g. Lua_Service_Wrapper loads project providing GUI visibility via mmShowGUI)
    std::function<void(bool)> provide_gui_visibility = [&](bool) {};
};

} /* end namespace frontend_resources */
} /* end namespace megamol */

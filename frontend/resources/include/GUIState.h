/*
 * GUIState.h
 *
 * Copyright (C) 2020 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

namespace megamol {
namespace frontend_resources {

struct GUIState {

    // Request GUI state: true  -> containing updated GUI state, GUI visibility and GUI scale already wrapped into respective lua commands
    //                    false -> containing the GUI state string only
    /// (e.g. ScreenshotService saves project to PNG header using complete LUA GUI state)
    std::function<std::string(bool)> request_gui_state = [&](bool) { return std::string(); };

    // Request GUI visibility
    std::function<bool(void)> request_gui_visibility = [&]() { return false; };

    // Request GUI scale
    std::function<float(void)> request_gui_scale = [&]() { return 1.0f; };

    // Provide GUI state as JSON string from argument of lua command
    /// (e.g. Lua_Service_Wrapper loads project providing GUI state via mmSetGUIState)
    std::function<void(const std::string&)> provide_gui_state;

    // Provide GUI visibility from argument of lua command
    /// (e.g. Lua_Service_Wrapper loads project providing GUI visibility via mmSetShowGUI)
    std::function<void(bool)> provide_gui_visibility;

    // Provide GUI scale from argument of lua command
    /// (e.g. Lua_Service_Wrapper loads project providing GUI scale via mmSetScaleGUI)
    std::function<void(float)> provide_gui_scale;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */

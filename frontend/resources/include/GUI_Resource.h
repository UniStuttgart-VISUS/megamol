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

    // Request string containing updated GUI state, GUI visibility and GUI scale already wrapped into respective lua commands
    /// (e.g. ScreenshotService saves project with GUI state tp PNG header)
    std::function<std::string(void)> request_gui_state = [&](void){ return std::string(); };

    // Provide GUI state as JSON string from argument of lua command
    /// (e.g. Lua_Service_Wrapper loads project providing GUI state via mmSetGUIState)
    std::function<void(std::string)> provide_gui_state;

    // Provide GUI visibility
    /// (e.g. Lua_Service_Wrapper loads project providing GUI visibility via mmShowGUI)
    std::function<void(bool)> provide_gui_visibility;

    // Provide GUI scale
    /// (e.g. Lua_Service_Wrapper loads project providing GUI scale via mmScaleGUI)
    std::function<void(float)> provide_gui_scale;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */

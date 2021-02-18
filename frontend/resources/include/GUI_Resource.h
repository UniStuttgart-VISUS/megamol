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

    // Function requesting updated GUI state (e.g. ScreenshotService save project with GUI state tp PNG header)
    std::function<std::string(void)> gui_state = nullptr;
};

} /* end namespace frontend_resources */
} /* end namespace megamol */

/*
 * WidgetPicking_gl.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED
#define MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED


#include "GUIUtils.h"

#include "mmcore/view/Input.h"

#include "vislib/sys/Log.h"


namespace megamol {
namespace gui {


/**
 * OpenGL implementation of widget picking.
 */
class WidgetPicking {
public:
    WidgetPicking(void);

    ~WidgetPicking(void) = default;

    void ProcessMouseMove(double x, double y);

    void ProcessMouseClick(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods);

    bool Enable(void);

    bool Disable(void);

private:
    // VARIABLES --------------------------------------------------------------
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED

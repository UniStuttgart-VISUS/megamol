/*
 * WidgetPicking_gl.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED
#define MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED


#include "GUIUtils.h"

#include "glowl/FramebufferObject.hpp"

#include "mmcore/view/Input.h"

#include "vislib/sys/Log.h"


namespace megamol {
namespace gui {


/**
 * OpenGL implementation of widget picking.
 *
 * Code adapted from megamol::mesh::Render3DUI
 *
 */
class WidgetPicking {
public:
    WidgetPicking(void);

    ~WidgetPicking(void);

    bool ProcessMouseMove(double x, double y);

    bool ProcessMouseClick(megamol::core::view::MouseButton button, megamol::core::view::MouseButtonAction action,
        megamol::core::view::Modifiers mods);

    bool Enable(void);

    bool Disable(void);

    static void DrawPickalbleCircle(unsigned int id);

private:
    // VARIABLES --------------------------------------------------------------

    double m_cursor_x, m_cursor_y;

    /**
     * Set to true if cursor on interactable object during current frame with respective obj id as second value
     * Set to fale false if cursor on "background" during current frame with -1 as second value
     */
    std::pair<bool, int> m_cursor_on_interaction_obj;

    /**
     * Set to true if  with respective obj id as second value
     * Set to fale false if cursor on "background" during current frame with -1 as second value
     */
    std::pair<bool, int> m_active_interaction_obj;

    std::unique_ptr<glowl::FramebufferObject> m_fbo;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WIDGETPICKING_GL_INCLUDED

/*
 * ParamPresentation.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETER_PRESENTATION_INCLUDED
#define MEGAMOL_GUI_PARAMETER_PRESENTATION_INCLUDED


#include "vislib/sys/Log.h"

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>


namespace megamol {
namespace gui {

/**
 * Manages GUI parameter presentations.
 */
class ParamPresentation {
public:
    ParamPresentation(void);

    ~ParamPresentation(void);

private:
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETER_PRESENTATION_INCLUDED
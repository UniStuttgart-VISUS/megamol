/*
 * ParameterOrbitalWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERORBITALWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERORBITALWIDGET_INCLUDED
#pragma once


#define IMGUIZMO_IMGUI_FOLDER
#include "glm/glm.hpp"
#include "imGuIZMOquat.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * Orbital parameter widget
 */
class ParameterOrbitalWidget {
public:
    ParameterOrbitalWidget();
    ~ParameterOrbitalWidget() = default;

    bool gizmo3D_rotation_axes(glm::vec4& inout_rotation);
    bool gizmo3D_rotation_direction(glm::vec3& inout_direction);

private:
    ::quat m_rotation;
    ::vec3 m_direction;

    bool init;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERORBITALWIDGET_INCLUDED

/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once


#define IMGUIZMO_IMGUI_FOLDER
#include "glm/glm.hpp"
#include "imGuIZMOquat.h"


namespace megamol::gui {


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


} // namespace megamol::gui

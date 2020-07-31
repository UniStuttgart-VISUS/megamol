/*
 * ParameterOrbitalWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParameterOrbitalWidget.h"


using namespace megamol;
using namespace megamol::gui;


ParameterOrbitalWidget::ParameterOrbitalWidget(void) : m_rotation(1.0f, 0.0f, 0.0f, 0.0f) {}


bool megamol::gui::ParameterOrbitalWidget::gizmo3D_direction(glm::vec4& inout_rotation) {
    bool retval = false;

    // this->m_rotation = static_cast<quat>(inout_rotation);

    if (ImGui::gizmo3D("##gizmo1", this->m_rotation, ImGui::GetNextItemWidth(), imguiGizmo::modeDirection)) {
        // inout_rotation = this->m_rotation;
        retval = true;
    }

    return retval;
}


bool megamol::gui::ParameterOrbitalWidget::gizmo3D_axes(glm::vec4& inout_rotation) {
    bool retval = false;

    // this->m_rotation = static_cast<quat>(inout_rotation);
    if (ImGui::gizmo3D("##gizmo1", this->m_rotation, ImGui::GetNextItemWidth(),
            imguiGizmo::mode3Axes | imguiGizmo::cubeAtOrigin | imguiGizmo::sphereAtOrigin)) {
        // inout_rotation = this->m_rotation;
        retval = true;
    }

    return retval;
}
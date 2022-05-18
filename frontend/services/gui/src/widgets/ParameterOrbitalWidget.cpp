/*
 * ParameterOrbitalWidget.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "ParameterOrbitalWidget.h"


using namespace megamol;
using namespace megamol::gui;


ParameterOrbitalWidget::ParameterOrbitalWidget()
        : m_rotation(1.0f, 0.0f, 0.0f, 0.0f)
        , m_direction(0.0f, 0.0f, 0.0f)
        , init(false) {}


bool megamol::gui::ParameterOrbitalWidget::gizmo3D_rotation_axes(glm::vec4& inout_rotation) {

    bool retval = false;
    if (this->init) {
        this->m_rotation = ::quat(inout_rotation.w, inout_rotation.x, inout_rotation.y, inout_rotation.z);
        this->init = true;
    }

    if (ImGui::gizmo3D("##gizmo_rotation", this->m_rotation, ImGui::CalcItemWidth(),
            imguiGizmo::mode3Axes | imguiGizmo::cubeAtOrigin | imguiGizmo::sphereAtOrigin)) {
        inout_rotation = glm::vec4(this->m_rotation.x, this->m_rotation.y, this->m_rotation.z, this->m_rotation.w);
        retval = true;
    } else {
        this->m_rotation = ::quat(inout_rotation.w, inout_rotation.x, inout_rotation.y, inout_rotation.z);
    }

    return retval;
}


bool megamol::gui::ParameterOrbitalWidget::gizmo3D_rotation_direction(glm::vec3& inout_direction) {

    bool retval = false;
    if (this->init) {
        this->m_direction = ::vec3(inout_direction.x, inout_direction.y, inout_direction.z);
        this->init = true;
    }

    if (ImGui::gizmo3D("##gizmo_direction", this->m_direction, ImGui::CalcItemWidth(), imguiGizmo::modeDirection)) {
        inout_direction = glm::vec3(this->m_direction.x, this->m_direction.y, this->m_direction.z);
        retval = true;
    } else {
        this->m_direction = ::vec3(inout_direction.x, inout_direction.y, inout_direction.z);
    }

    return retval;
}

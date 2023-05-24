/*
 * ParameterGroupClipPlaneWidget.h
 *
 * Copyright (C) 2023 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractParameterGroupWidget.h"
#include "mmcore/view/CameraSerializer.h"


namespace megamol::gui {


/** ************************************************************************
 * Clip plane widget for parameter group
 */
class ParameterGroupClipPlaneWidget : public AbstractParameterGroupWidget {
public:
    ParameterGroupClipPlaneWidget();
    ~ParameterGroupClipPlaneWidget() override = default;

    bool Check(ParamPtrVector_t& params) override;

    bool Draw(ParamPtrVector_t params, const std::string& in_search, megamol::gui::Parameter::WidgetScope in_scope,
        megamol::core::utility::PickingBuffer* inout_picking_buffer, ImGuiID in_override_header_state) override;

private:
    // FUNCTIONS --------------------------------------------------------------


    // TEMP:
    void OrthoGraphic(const float l, float r, float b, const float t, float zn, const float zf, float* m16);

    // VARIABLES --------------------------------------------------------------

    HoverToolTip tooltip;

    megamol::core::view::CameraSerializer cameraSerializer;

    glm::mat4 guizmo_mat;
};


} // namespace megamol::gui

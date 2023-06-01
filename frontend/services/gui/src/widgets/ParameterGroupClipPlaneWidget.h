/*
 * ParameterGroupClipPlaneWidget.h
 *
 * Copyright (C) 2023 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractParameterGroupWidget.h"
#include "mmcore/view/CameraSerializer.h"

#include <ImGuizmo.h>

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

    void drawPlane(const glm::mat4& mvp, float size, ImVec4 color, ImVec2 scree_pos, ImVec2 screen_size, bool plane_enabled);

    ImVec2 worldToPos(const glm::vec4& worldPos, const glm::mat4& mat, ImVec2 position, ImVec2 size);

    // VARIABLES --------------------------------------------------------------

    HoverToolTip tooltip;
    megamol::core::view::CameraSerializer camera_serializer;
    ImGuizmo::OPERATION guizmo_operation;

};


} // namespace megamol::gui

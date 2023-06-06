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

    void draw_plane(
        const glm::mat4& mvp, float size, ImVec4 color, ImVec2 screen_pos, ImVec2 screen_size, bool plane_enabled);

    void draw_grid(
        const glm::mat4& mvp, float size, ImVec4 color, ImVec2 screen_pos, ImVec2 screen_size, bool plane_enabled);

    ImVec2 world_to_screen(const glm::vec4& worldPos, const glm::mat4& mat, ImVec2 position, ImVec2 size);

    void widget_params(bool pop_up);

    // VARIABLES --------------------------------------------------------------

    HoverToolTip tooltip;
    megamol::core::view::CameraSerializer camera_serializer;
    ImGuizmo::OPERATION guizmo_operation;
    bool guizmo_draw_plane;
    bool guizmo_draw_grid;
};


} // namespace megamol::gui

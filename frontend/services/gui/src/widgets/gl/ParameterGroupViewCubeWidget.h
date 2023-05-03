/*
 * ParameterGroupViewCubeWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "../AbstractParameterGroupWidget.h"
#include "../ImageWidget.h"
#include "mmcore/view/CameraControllers.h"
#include "mmcore_gl/utility/RenderUtils.h"


namespace megamol::gui {

typedef core::utility::DefaultView DefaultView_t;
typedef core::utility::DefaultOrientation DefaultOrientation_t;


/** ***********************************************************************
 * Pickable Cube
 */
class PickableCube {
public:
    PickableCube();
    ~PickableCube() = default;

    bool Draw(unsigned int picking_id, int& inout_selected_face_id, int& inout_selected_orientation_id,
        int& out_hovered_face_id, int& out_hovered_orientation_id, const glm::vec4& cube_orientation,
        core::utility::ManipVector_t& pending_manipulations);

    core::utility::InteractVector_t GetInteractions(unsigned int id) const;

private:
    ImageWidget image_up_arrow;
    std::shared_ptr<glowl::GLSLProgram> shader;
};


/** ***********************************************************************
 * Pickable Texture
 */
class PickableTexture {
public:
    PickableTexture();
    ~PickableTexture() = default;

    bool Draw(unsigned int picking_id, int selected_face_id, int& out_orientation_change, int& out_hovered_arrow_id,
        megamol::core::utility::ManipVector_t& pending_manipulations);

    megamol::core::utility::InteractVector_t GetInteractions(unsigned int id) const;

private:
    ImageWidget image_rotation_arrow;
    std::shared_ptr<glowl::GLSLProgram> shader;
};


/** ***********************************************************************
 * View cube widget for parameter group
 */
class ParameterGroupViewCubeWidget : public AbstractParameterGroupWidget {
public:
    ParameterGroupViewCubeWidget();
    ~ParameterGroupViewCubeWidget() override = default;

    bool Check(bool only_check, ParamPtrVector_t& params) override;

    bool Draw(ParamPtrVector_t params, const std::string& in_search, megamol::gui::Parameter::WidgetScope in_scope,
        megamol::core::utility::PickingBuffer* inout_picking_buffer, ImGuiID in_override_header_state) override;

private:
    // VARIABLES --------------------------------------------------------------

    HoverToolTip tooltip;
    PickableCube cube_widget;
    PickableTexture texture_widget;

    megamol::core::param::AbstractParamPresentation::Presentation last_presentation;
};


} // namespace megamol::gui

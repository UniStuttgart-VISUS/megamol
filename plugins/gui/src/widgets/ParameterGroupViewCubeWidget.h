/*
 * ParameterGroupViewCubeWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED
#pragma once


#include "AbstractParameterGroupWidget.h"
#include "mmcore/view/AbstractView3D.h"


namespace megamol {
namespace gui {

    typedef megamol::core::view::AbstractView3D::DefaultView DefaultView_t;
    typedef megamol::core::view::AbstractView3D::DefaultOrientation DefaultOrientation_t;

    /** ***********************************************************************
     * Pickable Cube
     */
    class PickableCube {
    public:
        PickableCube(void);
        ~PickableCube(void) = default;

        bool Draw(unsigned int picking_id, int& inout_selected_face_id, int& inout_selected_orientation_id,
            int& out_hovered_face_id, int& out_hovered_orientation_id, const glm::vec4& cube_orientation,
            ManipVector& pending_manipulations);

        InteractVector GetInteractions(unsigned int id) const;

    private:
        ImageWidget image_up_arrow;
        std::shared_ptr<glowl::GLSLProgram> shader;
    };


    /** ***********************************************************************
     * Pickable Texture
     */
    class PickableTexture {
    public:
        PickableTexture(void);
        ~PickableTexture(void) = default;

        bool Draw(unsigned int picking_id, int selected_face_id, int& out_orientation_change, int& out_hovered_arrow_id,
            ManipVector& pending_manipulations);

        InteractVector GetInteractions(unsigned int id) const;

    private:
        ImageWidget image_rotation_arrow;
        std::shared_ptr<glowl::GLSLProgram> shader;
    };


    /** ***********************************************************************
     * View cube widget for parameter group.
     */
    class ParameterGroupViewCubeWidget : public AbstractParameterGroupWidget {
    public:
        ParameterGroupViewCubeWidget(void);

        ~ParameterGroupViewCubeWidget(void) = default;

        bool Check(bool only_check, ParamPtrVector_t& params);

        bool Draw(ParamPtrVector_t params, const std::string& in_search, megamol::gui::Parameter::WidgetScope in_scope,
            PickingBuffer* inout_picking_buffer);

    private:
        // VARIABLES --------------------------------------------------------------

        HoverToolTip tooltip;
        PickableCube cube_widget;
        PickableTexture texture_widget;

        megamol::core::param::AbstractParamPresentation::Presentation last_presentation;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED

/*
 * ParameterGroupViewCubeWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED


#include "AbstractParameterGroupWidget.h"
#include "ImageWidget_gl.h"
#include "mmcore/view/RenderUtils.h"


namespace megamol {
namespace gui {


    /** ***********************************************************************
     * Pickable Cube
     */
    class PickableCube {
    public:
        PickableCube(void);
        ~PickableCube(void) = default;

        bool Draw(unsigned int picking_id, int& inout_face_id, int& inout_orientation_id, int& out_hovered_face_id,
            int& out_hovered_orientation_id, const glm::vec4& cube_orientation, ManipVector& pending_manipulations);

        InteractVector GetInteractions(unsigned int id) const;

    private:
        ImageWidget image_up_arrow;
        std::shared_ptr<glowl::GLSLProgram> shader;

        /// TEMP
        enum Corners : int  {
            CORNER_NONE          = 0,
            CORNER_TOP_LEFT      = 1 << 0,
            CORNER_TOP_RIGHT     = 1 << 1,
            CORNER_BOTTOM_LEFT   = 1 << 2,
            CORNER_BOTTOM_RIGHT  = 1 << 3
        };
        enum Edges : int  {
            EDGE_NONE          = 0,
            EDGE_TOP_LEFT      = 1 << 0,
            EDGE_TOP_RIGHT     = 1 << 1,
            EDGE_BOTTOM_LEFT   = 1 << 2,
            EDGE_BOTTOM_RIGHT  = 1 << 3
        };
        Edges   edge_hover_id;
        Corners corner_hover_id;
    };


    /** ***********************************************************************
     * Pickable Texture
     */
    class PickableTexture {
    public:
        PickableTexture(void);
        ~PickableTexture(void) = default;

        bool Draw(unsigned int picking_id, int face_id, int& out_orientation_change, int& out_hovered_arrow_id,
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

        bool Draw(ParamPtrVector_t params, const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::Parameter::WidgetScope in_scope, PickingBuffer* inout_picking_buffer);

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

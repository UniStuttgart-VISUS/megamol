/*
 * ParameterGroupViewCubeWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED


#include "ImageWidget_gl.h"
#include "AbstractParameterGroupWidget.h"
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

        bool Draw(unsigned int id, int& inout_face_index, int& inout_orientation_index,
                  int& out_hovered_face, int& out_hovered_orientation, const glm::vec4& cube_orientation, ManipVector& pending_manipulations);

        InteractVector GetInteractions(unsigned int id) const;

    private:
        std::shared_ptr<glowl::GLSLProgram> shader;
    };


    /** ***********************************************************************
     * Pickable Texture
     */
    class PickableTexture {
    public:
        PickableTexture(void);
        ~PickableTexture(void) = default;

        bool Draw(unsigned int id, int& out_orientation_index_offset, int& out_hovered_arrow, ManipVector& pending_manipulations);

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

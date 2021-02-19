/*
 * ParameterGroupViewCubeWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED


#include "AbstractParameterGroupWidget.h"
#include "mmcore/view/RenderUtils.h"

namespace megamol {
namespace gui {


    /**
     * Pickable Cube
     */
    class PickableCube {
    public:
        PickableCube(void);
        ~PickableCube(void) = default;

        bool Draw(unsigned int id, int& inout_view_index, int& inout_orientation_index, int& out_view_hover_index,
            int& out_orientation_hover_index, const glm::vec4& view_orientation, const glm::vec2& vp_dim,
            ManipVector& pending_manipulations);

        InteractVector GetInteractions(unsigned int id) const;

    private:
        std::shared_ptr<glowl::GLSLProgram> shader;
        megamol::core::view::RenderUtils render_utils;
    };


    /**
     * View cube widget for parameter group.
     */
    class ParameterGroupViewCubeWidget : public AbstractParameterGroupWidget {
    public:
        ParameterGroupViewCubeWidget(void);

        ~ParameterGroupViewCubeWidget(void) = default;

        bool Check(bool only_check, ParamPtrVector_t& params);

        bool Draw(ParamPtrVector_t params, const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::ParameterPresentation::WidgetScope in_scope, PickingBuffer* inout_picking_buffer);

    private:
        // VARIABLES --------------------------------------------------------------

        HoverToolTip tooltip;
        PickableCube cube_widget;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED

/*
 * ParameterGroupViewCubeWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPVIEWCUBEWIDGET_INCLUDED


#include "AbstractParameterGroupWidget.h"


namespace megamol {
namespace gui {


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

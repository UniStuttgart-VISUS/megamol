/*
 * ParameterGroupAnimationWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPANIMATIONWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPANIMATIONWIDGET_INCLUDED


#include "AbstractParameterGroupWidget.h"


namespace megamol {
namespace gui {


    /**
     * Animation widget for parameter group.
     */
    class ParameterGroupAnimationWidget : public AbstractParameterGroupWidget {
    public:
        ParameterGroupAnimationWidget(void);

        ~ParameterGroupAnimationWidget(void) = default;

        bool Check(bool only_check, ParamPtrVector_t& params);

        bool Draw(ParamPtrVector_t params, const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::ParameterPresentation::WidgetScope in_scope, PickingBuffer* inout_picking_buffer);

    private:
        // VARIABLES --------------------------------------------------------------

        ImVec2 speed_knob_pos;
        ImVec2 time_knob_pos;
        struct {
            ImageWidget play;
            ImageWidget pause;
            ImageWidget fastforward;
            ImageWidget fastrewind;
        } image_buttons;

        HoverToolTip tooltip;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPANIMATIONWIDGET_INCLUDED

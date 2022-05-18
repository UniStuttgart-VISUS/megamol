/*
 * ParameterGroupAnimationWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPANIMATIONWIDGET_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPANIMATIONWIDGET_INCLUDED
#pragma once


#include "AbstractParameterGroupWidget.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * Animation widget for parameter group
 */
class ParameterGroupAnimationWidget : public AbstractParameterGroupWidget {
public:
    ParameterGroupAnimationWidget();
    ~ParameterGroupAnimationWidget() override = default;

    bool Check(bool only_check, ParamPtrVector_t& params) override;

    bool Draw(ParamPtrVector_t params, const std::string& in_search, megamol::gui::Parameter::WidgetScope in_scope,
        megamol::core::utility::PickingBuffer* inout_picking_buffer, ImGuiID in_override_header_state) override;

private:
    // VARIABLES --------------------------------------------------------------

    struct {
        ImageWidget play_pause;
        ImageWidget fastforward;
        ImageWidget fastrewind;
    } image_buttons;

    HoverToolTip tooltip;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPANIMATIONWIDGET_INCLUDED

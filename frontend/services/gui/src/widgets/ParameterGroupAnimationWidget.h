/*
 * ParameterGroupAnimationWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractParameterGroupWidget.h"


namespace megamol::gui {


/** ************************************************************************
 * Animation widget for parameter group
 */
class ParameterGroupAnimationWidget : public AbstractParameterGroupWidget {
public:
    ParameterGroupAnimationWidget();
    ~ParameterGroupAnimationWidget() override = default;

    bool Check(ParamPtrVector_t& params) override;

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


} // namespace megamol::gui

/*
 * HoverToolTip.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_HOVERTOOLTIP_INCLUDED
#define MEGAMOL_GUI_HOVERTOOLTIP_INCLUDED
#pragma once


#include "gui_utils.h"
#include <string>


namespace megamol {
namespace gui {


/** ************************************************************************
 * Hover tooltip widget
 */
class HoverToolTip {
public:
    HoverToolTip();
    ~HoverToolTip() = default;

    /**
     * Draw tooltip on hover.
     *
     * @param text        The tooltip text.
     * @param id          The id of the imgui item the tooltip belongs (only needed for delayed appearance of
     * tooltip).
     * @param time_start  The time delay to wait until the tooltip is shown for a hovered imgui item.
     * @param time_end    The time delay to wait until the tooltip is hidden for a hovered imgui item.
     */
    bool ToolTip(const std::string& text, ImGuiID id = 0, float time_start = 0.0f, float time_end = 4.0f);

    /**
     * Show help marker text with tooltip on hover.
     *
     * @param text   The help tooltip text.
     * @param label  The visible text for which the tooltip is enabled.
     */
    bool Marker(const std::string& text, const std::string& label = "(?)");

    /**
     * Reset toopltip time and widget id.
     */
    void Reset();

private:
    // VARIABLES --------------------------------------------------------------

    float tooltip_time;
    ImGuiID tooltip_id;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_HOVERTOOLTIP_INCLUDED

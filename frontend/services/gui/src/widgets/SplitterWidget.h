/*
 * SplitterWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "gui_utils.h"
#include <string>


namespace megamol::gui {


/** ************************************************************************
 * Splitter widget
 */
class SplitterWidget {
public:
    SplitterWidget();
    ~SplitterWidget() = default;

    enum FixedSplitterSide { LEFT_TOP, RIGHT_BOTTOM };

    /**
     * Draw draggable splitter between two child windows, relative to parent window size.
     * https://github.com/ocornut/imgui/issues/319
     *
     * @param idstr                         The identifier string for the splitter widget
     * @param vertical                      If true draw vertical splitter, if false draw horizontal splitter
     * @param length                        The length of the splitter
     * @param fixed_side                    Define which side of the splitter has fixed width/height. Set to zero for autodetection.
     * @param inout_range_left_top          The returned size of the respective side
     * @param inout_range_right_bottom      The returned size of the respective side
     * @param window_cursor_pos             The upper left position of the parent window
     */
    bool Widget(const std::string& idstr, bool vertical, float length, FixedSplitterSide fixed_side,
        float& inout_range_left_top, float& inout_range_right_bottom, ImVec2 window_cursor_pos);

    float GetWidth() const;

private:
    // VARIABLES --------------------------------------------------------------

    /** Splitter width for restoring after collapsing.  */
    float splitter_last_width;
};


} // namespace megamol::gui

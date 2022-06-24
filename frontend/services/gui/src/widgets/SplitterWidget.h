/*
 * SplitterWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_SPLITTERWIDGET_INCLUDED
#define MEGAMOL_GUI_SPLITTERWIDGET_INCLUDED
#pragma once


#include <string>


namespace megamol {
namespace gui {


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
     */
    bool Widget(const std::string& idstr, bool vertical, float length, FixedSplitterSide fixed_side,
        float& inout_range_left_top, float& inout_range_right_bottom);

    float GetWidth() const;

private:
    // VARIABLES --------------------------------------------------------------

    /** Splitter width for restoring after collapsing.  */
    float splitter_last_width;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_SPLITTERWIDGET_INCLUDED

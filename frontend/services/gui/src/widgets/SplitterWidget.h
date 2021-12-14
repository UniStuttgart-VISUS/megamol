/*
 * SplitterWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_SPLITTERWIDGET_INCLUDED
#define MEGAMOL_GUI_SPLITTERWIDGET_INCLUDED
#pragma once


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
     * @param split_vertically      The
     * @param fixed_side            The
     * @param size_left_top         The
     * @param size_right_bottom     The
     */
    bool Widget(bool split_vertically, FixedSplitterSide fixed_side, float& size_left_top, float& size_right_bottom);

private:
    // VARIABLES --------------------------------------------------------------

    /** Splitter width for restoring after collapsing.  */
    float splitter_last_width;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_SPLITTERWIDGET_INCLUDED

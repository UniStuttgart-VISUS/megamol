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


    /**
     * Splitter widget.
     */
    class SplitterWidget {
    public:
        SplitterWidget(void);

        ~SplitterWidget(void) = default;

        enum FixedSplitterSide { LEFT, RIGHT };

        /**
         * Draw draggable splitter between two child windows, relative to parent window size.
         * https://github.com/ocornut/imgui/issues/319
         */
        bool Widget(FixedSplitterSide fixed_side, float& size_left, float& size_right);

    private:
        // VARIABLES --------------------------------------------------------------

        /** Splitter width for restoring after collapsing.  */
        float splitter_last_width;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_SPLITTERWIDGET_INCLUDED

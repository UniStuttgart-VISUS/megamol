/*
 * StringSearchWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_STRINGSEARCHWIDGET_INCLUDED
#define MEGAMOL_GUI_STRINGSEARCHWIDGET_INCLUDED
#pragma once


#include "HoverToolTip.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * String search widget
 */
class StringSearchWidget {
public:
    StringSearchWidget();
    ~StringSearchWidget() = default;

    bool Widget(const std::string& label, const std::string& help, bool omit_focus = false);

    inline void SetSearchFocus() {
        this->search_focus = 2;
    }

    inline std::string GetSearchString() const {
        return this->search_string;
    }

    inline void ClearSearchString() {
        this->search_string.clear();
    }

private:
    // VARIABLES --------------------------------------------------------------

    unsigned int search_focus;
    std::string search_string;
    HoverToolTip tooltip;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_STRINGSEARCHWIDGET_INCLUDED

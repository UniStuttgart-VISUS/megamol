/*
 * StringSearchWidget.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_STRINGSEARCHWIDGET_INCLUDED
#define MEGAMOL_GUI_STRINGSEARCHWIDGET_INCLUDED


#include "GUIUtils.h"
#include "HoverToolTip.h"


namespace megamol {
namespace gui {


/**
 * String search widget.
 */
class StringSearchWidget {
public:
    StringSearchWidget(void);

    ~StringSearchWidget(void) = default;

    /**
     * Returns true if search string is found in source as a case insensitive substring.
     *
     * @param source   The string to search in.
     * @param search   The string to search for in the source.
     */
    static bool FindCaseInsensitiveSubstring(const std::string& source, const std::string& search) {
        if (search.empty()) return true;
        auto it = std::search(source.begin(), source.end(), search.begin(), search.end(),
            [](char ch1, char ch2) { return std::toupper(ch1) == std::toupper(ch2); });
        return (it != source.end());
    }

    bool Widget(const std::string& label, const std::string& help);

    inline void SetSearchFocus(bool focus) { this->search_focus = focus; }

    inline std::string GetSearchString(void) const { return this->search_string; }

private:
    // VARIABLES --------------------------------------------------------------

    bool search_focus;
    std::string search_string;
    HoverToolTip tooltip;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_STRINGSEARCHWIDGET_INCLUDED

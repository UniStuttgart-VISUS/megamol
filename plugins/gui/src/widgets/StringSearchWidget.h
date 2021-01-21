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

        bool Widget(const std::string& label, const std::string& help, bool apply_focus = true);

        inline void SetSearchFocus(bool focus) {
            this->search_focus = focus;
        }

        inline std::string GetSearchString(void) const {
            return this->search_string;
        }

        inline void ClearSearchString(void) {
            this->search_string.clear();
        }

    private:
        // VARIABLES --------------------------------------------------------------

        bool search_focus;
        std::string search_string;
        HoverToolTip tooltip;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_STRINGSEARCHWIDGET_INCLUDED

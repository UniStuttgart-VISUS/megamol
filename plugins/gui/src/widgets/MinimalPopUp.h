/*
 * MinimalPopUp.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_MINIMALPOPUP_INCLUDED
#define MEGAMOL_GUI_MINIMALPOPUP_INCLUDED


#include "GUIUtils.h"


namespace megamol {
namespace gui {


/**
 * String search widget.
 */
class MinimalPopUp {
public:
    static bool PopUp(const std::string& label_id, bool open_popup, const std::string& info_text,
        const std::string& confirm_btn_text, bool& confirmed, const std::string& abort_btn_text, bool& aborted);

private:
    MinimalPopUp(void) = default;

    ~MinimalPopUp(void) = default;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_MINIMALPOPUP_INCLUDED

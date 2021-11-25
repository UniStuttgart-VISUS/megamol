/*
 * PopUps.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_MINIMALPOPUP_INCLUDED
#define MEGAMOL_GUI_MINIMALPOPUP_INCLUDED
#pragma once


#include "HoverToolTip.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * Different pup-up widgets
 */
class PopUps {
public:
    PopUps();
    ~PopUps() = default;

    // Rename pop-up
    bool Rename(const std::string& caption, bool open_popup, std::string& rename);

    // STATIC Minimal Popup providing feedback of up to two buttons
    static bool Minimal(const std::string& label_id, bool open_popup, const std::string& info_text,
        const std::string& confirm_btn_text, bool& confirmed, const std::string& abort_btn_text, bool& aborted);

    static bool Minimal(const std::string& label_id, bool open_popup, const std::string& info_text,
        const std::string& confirm_btn_text, bool& confirmed) {
        return Minimal(label_id, open_popup, info_text, confirm_btn_text, confirmed, "", confirmed);
    }

    static bool Minimal(const std::string& label_id, bool open_popup, const std::string& info_text) {
        bool confirmed;
        return Minimal(label_id, open_popup, info_text, "", confirmed, "", confirmed);
    }

    static bool Minimal(const std::string& label_id, bool open_popup, const std::string& info_text,
        const std::string& confirm_btn_text) {
        bool confirmed;
        return Minimal(label_id, open_popup, info_text, confirm_btn_text, confirmed, "", confirmed);
    }

private:
    std::string rename_string;
    HoverToolTip rename_tooltip;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_MINIMALPOPUP_INCLUDED

/*
 * RenamePopUp.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_RENAMEPOPUP_INCLUDED
#define MEGAMOL_GUI_RENAMEPOPUP_INCLUDED


#include "GUIUtils.h"


namespace megamol {
namespace gui {


/**
 * String search widget.
 */
class RenamePopUp {
public:
    RenamePopUp(void);

    ~RenamePopUp(void) = default;

    bool PopUp(const std::string& caption, bool open_popup, std::string& rename);

private:
    std::string rename_string;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_RENAMEPOPUP_INCLUDED

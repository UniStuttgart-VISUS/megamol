/*
 * HotkeyEditor.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_HOTKEYEDITOR_H_INCLUDED
#define MEGAMOL_GUI_HOTKEYEDITOR_H_INCLUDED
#pragma once


#include "AbstractWindow.h"


namespace megamol {
namespace gui {

    class HotkeyEditor : public AbstractWindow {
    public:
        explicit HotkeyEditor(const std::string& window_name);
        ~HotkeyEditor();

        bool Update() override;
        bool Draw() override;

        void SpecificStateFromJSON(const nlohmann::json& in_json) override;
        void SpecificStateToJSON(nlohmann::json& inout_json) override;

    private:
        // VARIABLES --------------------------------------------------------------

    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_HOTKEYEDITOR_H_INCLUDED

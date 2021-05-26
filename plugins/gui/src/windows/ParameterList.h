/*
 * ParameterList.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED
#define MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED
#pragma once


#include "WindowConfiguration.h"
#include "widgets/StringSearchWidget.h"


namespace megamol {
namespace gui {

    /*
     * The parameter list GUI window.
     */
    class ParameterList : public WindowConfiguration {
    public:

        ParameterList(const std::string& window_name);
        ~ParameterList() = default;

        bool Update() override;
        bool Draw() override;
        void PopUps() override;

        void SpecificStateFromJSON(const nlohmann::json& in_json) override;
        void SpecificStateToJSON(nlohmann::json& inout_json) override;

    private:
        // VARIABLES --------------------------------------------------------------

        bool win_show_hotkeys;                      // [SAVED] flag to toggle showing only parameter hotkeys
        std::vector<std::string> win_modules_list;  // [SAVED] modules to show in a parameter window (show all if empty)
        bool win_extended_mode;                     // [SAVED] flag toggling between Expert and Basic parameter mode.

        // Widgets
        StringSearchWidget search_widget;
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool consider_module(const std::string& modname, std::vector<std::string>& modules_list);

    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED

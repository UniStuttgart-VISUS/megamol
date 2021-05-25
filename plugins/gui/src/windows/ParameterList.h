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
     * The ...
     */
    class ParameterList : public WindowConfiguration {
    public:

        ParameterList();
        ~ParameterList();

        void Update() override;

        void Draw() override;

        bool SpecificStateFromJSON(const nlohmann::json& in_json) override;

        bool SpecificStateToJSON(nlohmann::json& inout_json) override;

    private:
        // VARIABLES --------------------------------------------------------------

        StringSearchWidget search_widget;

        bool param_show_hotkeys = false;             // [SAVED] flag to toggle showing only parameter hotkeys
        std::vector<std::string> param_modules_list; // [SAVED] modules to show in a parameter window (show all if empty)
        bool param_extended_mode = false;            // [SAVED] flag toggling between Expert and Basic parameter mode.

        // FUNCTIONS --------------------------------------------------------------


    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED

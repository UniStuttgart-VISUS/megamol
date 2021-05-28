/*
 * ParameterList.h
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED
#define MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED
#pragma once


#include "AbstractWindow.h"
#include "widgets/StringSearchWidget.h"
#include "windows/Configurator.h"
#include "windows/TransferFunctionEditor.h"


namespace megamol {
namespace gui {


    /* ************************************************************************
     * The parameter list GUI window
     */
    class ParameterList : public AbstractWindow {
    public:
        explicit ParameterList(const std::string& window_name);
        ~ParameterList() = default;

        // Call once
        void SetData(std::shared_ptr<Configurator>& win_configurator,
            std::shared_ptr<TransferFunctionEditor>& win_tfeditor,
            const std::function<void(const std::string& window_name)>& add_window) {
            this->win_configurator_ptr = win_configurator;
            this->win_tfeditor_ptr = win_tfeditor;
            this->add_window_func = add_window;
        }
        bool Update() override;
        bool Draw() override;
        void PopUps() override;

        void SpecificStateFromJSON(const nlohmann::json& in_json) override;
        void SpecificStateToJSON(nlohmann::json& inout_json) override;

    private:
        // VARIABLES --------------------------------------------------------------

        /** Shortcut pointer to other windows */
        std::shared_ptr<Configurator> win_configurator_ptr;
        std::shared_ptr<TransferFunctionEditor> win_tfeditor_ptr;

        std::function<void(const std::string& window_name)> add_window_func;

        bool win_show_param_hotkeys;               // [SAVED] flag to toggle showing only parameter hotkeys
        std::vector<std::string> win_modules_list; // [SAVED] modules to show in a parameter window (show all if empty)
        bool win_extended_mode;                    // [SAVED] flag toggling between Expert and Basic parameter mode.

        // Widgets
        StringSearchWidget search_widget;
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        bool consider_module(const std::string& modname, std::vector<std::string>& modules_list) const;
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED

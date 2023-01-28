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


namespace megamol::gui {


/* ************************************************************************
 * The parameter list GUI window
 */
class ParameterList : public AbstractWindow {
public:
    typedef std::function<void(const std::string&, AbstractWindow::WindowConfigID, ImGuiID)>
        RequestParamWindowCallback_t;

    ParameterList(const std::string& window_name, AbstractWindow::WindowConfigID win_id, ImGuiID initial_module_uid,
        std::shared_ptr<Configurator> win_configurator, std::shared_ptr<TransferFunctionEditor> win_tfeditor,
        const RequestParamWindowCallback_t& add_parameter_window);
    ~ParameterList() = default;

    bool Update() override;
    bool Draw() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

private:
    // VARIABLES --------------------------------------------------------------

    /** Shortcut pointer to other windows */
    std::shared_ptr<Configurator> win_configurator_ptr;
    std::shared_ptr<TransferFunctionEditor> win_tfeditor_ptr;
    RequestParamWindowCallback_t request_new_parameter_window_func;

    std::vector<ImGuiID> win_modules_list; // [SAVED] modules to show in a parameter window (show all if empty)
    bool win_extended_mode;                // [SAVED] flag toggling between Expert and Basic parameter mode.

    // Widgets
    StringSearchWidget search_widget;
    HoverToolTip tooltip;
};

} // namespace megamol::gui

#endif // MEGAMOL_GUI_PARAMETERLIST_H_INCLUDED

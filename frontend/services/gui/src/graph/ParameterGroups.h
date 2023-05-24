/*
 * ParameterGroups.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "graph/Parameter.h"
#include "vislib/math/Ternary.h"
#include "widgets/AbstractParameterGroupWidget.h"
#include "widgets/HoverToolTip.h"


namespace megamol::gui {


/** ************************************************************************
 * Defines parameter widget groups depending on parameter namespaces
 */
class ParameterGroups {
public:
    // STATIC functions ---------------------------------------------------

    static void DrawParameter(megamol::gui::Parameter& inout_param, const std::string& in_search,
        megamol::gui::Parameter::WidgetScope in_scope, std::shared_ptr<TransferFunctionEditor> tfeditor_ptr);

    static void DrawGroupedParameters(const std::string& in_group_name,
        AbstractParameterGroupWidget::ParamPtrVector_t& params, const std::string& in_search,
        megamol::gui::Parameter::WidgetScope in_scope, std::shared_ptr<TransferFunctionEditor> tfeditor_ptr,
        ImGuiID in_override_header_state);

    // --------------------------------------------------------------------

    ParameterGroups();
    ~ParameterGroups() = default;

    bool Draw(megamol::gui::ParamVector_t& inout_params, const std::string& in_search, bool in_extended, bool in_indent,
        megamol::gui::Parameter::WidgetScope in_scope, std::shared_ptr<TransferFunctionEditor> tfeditor_ptr,
        ImGuiID in_override_header_state, megamol::core::utility::PickingBuffer* inout_picking_buffer);

    bool StateFromJSON(const nlohmann::json& in_json, const std::string& module_fullname);
    bool StateToJSON(nlohmann::json& inout_json, const std::string& module_fullname);

    bool ParametersVisible(megamol::gui::ParamVector_t& in_params);

private:
    typedef std::vector<megamol::gui::Parameter*> ParamPtrVector_t;
    typedef std::map<std::string, ParamPtrVector_t> ParamGroup_t;

    // VARIABLES --------------------------------------------------------------

    HoverToolTip tooltip;

    std::vector<std::shared_ptr<AbstractParameterGroupWidget>> group_widgets;
};


} // namespace megamol::gui

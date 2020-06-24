/*
 * ParameterGroupPresentation.cpp
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "ParameterGroupPresentation.h"


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::gui::configurator;


megamol::gui::ParameterGroupPresentation::ParameterGroupPresentation(void) {}


megamol::gui::ParameterGroupPresentation::~ParameterGroupPresentation(void) {}


bool megamol::gui::ParameterGroupPresentation::PresentGUI(megamol::gui::configurator::ParamVectorType& inout_params,
    const std::string& in_module_fullname, const std::string& in_search, bool in_extended, bool in_ignore_extended,
    megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
    const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor) {

    out_open_external_tf_editor = false;
    std::map<std::string, std::vector<megamol::gui::configurator::Parameter*>> group_map;

    // Sort parameters to groups
    for (auto& param : inout_params) {
        auto param_namespace = param.GetNameSpace();
        if (!param_namespace.empty()) {
            group_map[param_namespace].emplace_back(&param);
        }
    }

    // Draw parameters
    for (auto& param : inout_params) {
        auto param_name = param.full_name;
        auto param_namespace = param.GetNameSpace();

        if (param_namespace.empty()) {
            if ((param.type == ParamType::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
                param.present.ConnectExternalTransferFunctionEditor(in_external_tf_editor);
            }
            if (in_scope == megamol::gui::configurator::ParameterPresentation::WidgetScope::LOCAL) {
                bool param_searched = megamol::gui::GUIUtils::FindCaseInsensitiveSubstring(param_name, in_search);
                if (param_searched) {
                    if (!in_ignore_extended) {
                        param.present.extended = in_extended;
                    }
                    if (param.PresentGUI(in_scope)) {

                        // Open window calling the transfer function editor callback
                        if ((param.type == ParamType::TRANSFERFUNCTION) && (in_external_tf_editor != nullptr)) {
                            out_open_external_tf_editor = true;
                            auto param_fullname = std::string(in_module_fullname.c_str()) + "::" + param.full_name;
                            in_external_tf_editor->SetConnectedParameter(&param, param_fullname);
                        }
                    }
                }
            } else { /// scope == megamol::gui::configurator::ParameterPresentation::WidgetScope::GLOBAL
                param.PresentGUI(in_scope);
            }
        }
    }

    return true;
}
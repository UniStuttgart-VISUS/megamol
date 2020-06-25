/*
 * ParameterGroupPresentation.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPPRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPPRESENTATION_H_INCLUDED


#include "GUIUtils.h"
#include "configurator/Parameter.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * Defines parameter widget groups depending on parameter namespaces.
 */
class ParameterGroupPresentation {
public:
    // FUCNTIONS --------------------------------------------------------------


    ParameterGroupPresentation(void);
    ~ParameterGroupPresentation(void);

    bool PresentGUI(megamol::gui::configurator::ParamVectorType& inout_params, const std::string& in_module_fullname,
        const std::string& in_search, bool in_extended, bool in_ignore_extended,
        megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor);


private:
    typedef std::map<megamol::gui::ParamType, unsigned int> GroupWidgetType;
    typedef std::vector<megamol::gui::configurator::Parameter*> ParamPtrVectorType;
    typedef std::function<void(ParamPtrVectorType& params)> GroupWidgetCallbackFunc;
    typedef std::map<std::string, std::pair<std::vector<megamol::gui::configurator::Parameter*>, GroupWidgetType>>
        ParamGroupType;

    // VARIABLES --------------------------------------------------------------

    std::map<std::string, std::pair<GroupWidgetType, GroupWidgetCallbackFunc>> group_widget_ids;

    // FUCNTIONS --------------------------------------------------------------

    void drawParameter(megamol::gui::configurator::Parameter& inout_param, const std::string& in_module_fullname,
        const std::string& in_search, bool in_extended, bool in_ignore_extended,
        megamol::gui::configurator::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor);

    void group_widget_animation(ParamPtrVectorType& params);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPPRESENTATION_H_INCLUDED

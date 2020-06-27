/*
 * ParameterGroups.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

/* HOWTO add your own parameter group:
 * 1] Add new 'ParamType' for new widget group in core/param/AbstractParamPresentation.h
 * 2] Add new 'Presentation' for new widget group in core/param/AbstractParamPresentation.h (and add name to
 * presentation_name_map in CTOR) 3] Add 'Presentation' to ParamType in AbstractParamPresentation::InitPresentation() 4]
 * Add function drawing the new group widget: void group_widget_NEW-NAME(ParamPtrVectorType& params,
 * megamol::core::param::AbstractParamPresentation::Presentation presentation); 5] Add group widget data in CTOR:
 * GroupWidgetData NEW_NAME and add new function as callback
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED


#include "GUIUtils.h"
#include "graph/Parameter.h"

#include "mmcore/param/AbstractParamPresentation.h"


namespace megamol {
namespace gui {


/** ************************************************************************
 * Defines parameter widget groups depending on parameter namespaces.
 */
class ParameterGroups {
public:
    // FUCNTIONS --------------------------------------------------------------

    ParameterGroups(void);
    ~ParameterGroups(void);

    bool PresentGUI(megamol::gui::ParamVectorType& inout_params, const std::string& in_module_fullname,
        const std::string& in_search, bool in_extended, bool in_ignore_extended,
        megamol::gui::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor);

    bool ParameterGroupGUIStateFromJSONString(const std::string& in_json_string, const std::string& module_fullname);
    bool ParameterGroupGUIStateToJSON(nlohmann::json& inout_json, const std::string& module_fullname);


private:
    typedef std::vector<megamol::gui::Parameter*> ParamPtrVectorType;
    typedef std::map<megamol::gui::ParamType, unsigned int> GroupWidgetType;
    typedef std::map<std::string, std::pair<ParamPtrVectorType, GroupWidgetType>> ParamGroupType;
    typedef std::function<bool(ParamPtrVectorType& params,
        megamol::core::param::AbstractParamPresentation::Presentation presentation,
        megamol::gui::ParameterPresentation::WidgetScope in_scope)>
        GroupWidgetCallbackFunc;


    // Data needed for group widgets
    class GroupWidgetData : public megamol::core::param::AbstractParamPresentation {
    public:
        GroupWidgetData(void) { this->InitPresentation(AbstractParamPresentation::ParamType::GROUP_ANIMATION); }
        ~GroupWidgetData(void) {}
        bool active;
        GroupWidgetType type;
        GroupWidgetCallbackFunc callback;
    };

    // VARIABLES --------------------------------------------------------------

    GUIUtils utils;
    std::map<std::string, GroupWidgetData> group_widget_ids;

    struct {
        GLuint play;
        GLuint pause;
        GLuint fastforward;
        GLuint fastrewind;
    } button_tex_ids;

    // FUCNTIONS --------------------------------------------------------------

    void draw_parameter(megamol::gui::Parameter& inout_param, const std::string& in_module_fullname,
        const std::string& in_search, megamol::gui::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor);

    void draw_grouped_parameters(const std::string& in_group_name, ParamPtrVectorType& params,
        const std::string& in_module_fullname, const std::string& in_search,
        megamol::gui::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool& out_open_external_tf_editor);

    bool group_widget_animation(ParamPtrVectorType& params,
        megamol::core::param::AbstractParamPresentation::Presentation presentation,
        megamol::gui::ParameterPresentation::WidgetScope in_scope);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED

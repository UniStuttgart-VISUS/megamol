/*
 * ParameterGroups.h
 *
 * Copyright (C) 2020 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED
#define MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED


#include "GUIUtils.h"
#include "graph/Parameter.h"
#include "widgets/HoverToolTip.h"
#include "widgets/ImageWidget_gl.h"

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
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor);

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

    std::map<std::string, GroupWidgetData> group_widget_ids;

    // Widgets
    HoverToolTip tooltip;

    // ANIM group widget
    ImVec2 speed_knob_pos;
    ImVec2 time_knob_pos;
    struct {
        ImageWidget play;
        ImageWidget pause;
        ImageWidget fastforward;
        ImageWidget fastrewind;
    } image_buttons;

    // FUCNTIONS --------------------------------------------------------------

    void draw_parameter(megamol::gui::Parameter& inout_param, const std::string& in_module_fullname,
        const std::string& in_search, megamol::gui::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor);

    void draw_grouped_parameters(const std::string& in_group_name, ParamPtrVectorType& params,
        const std::string& in_module_fullname, const std::string& in_search,
        megamol::gui::ParameterPresentation::WidgetScope in_scope,
        const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor);

    bool group_widget_animation(ParamPtrVectorType& params,
        megamol::core::param::AbstractParamPresentation::Presentation presentation,
        megamol::gui::ParameterPresentation::WidgetScope in_scope);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED

/*
 * ParameterGroupsPresentation.h
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
#include "widgets/ParameterGroupAnimationWidget.h"
#include "widgets/ParameterGroupViewCubeWidget.h"
#include "widgets/WidgetPicking_gl.h"

#include "vislib/math/Ternary.h"


namespace megamol {
namespace gui {


    /**
     * Defines parameter widget groups depending on parameter namespaces.
     */
    class ParameterGroupsPresentation {
    public:
        // FUCNTIONS --------------------------------------------------------------

        ParameterGroupsPresentation(void);
        ~ParameterGroupsPresentation(void);

        bool PresentGUI(megamol::gui::ParamVector_t& inout_params, const std::string& in_module_fullname,
            const std::string& in_search, vislib::math::Ternary in_extended, bool in_indent,
            megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer);

        bool StateFromJSON(const nlohmann::json& in_json, const std::string& module_fullname);
        bool StateToJSON(nlohmann::json& inout_json, const std::string& module_fullname);

        static void DrawParameter(megamol::gui::Parameter& inout_param, const std::string& in_module_fullname,
            const std::string& in_search, megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor);

        static void DrawGroupedParameters(const std::string& in_group_name,
            AbstractParameterGroupWidget::ParamPtrVector_t& params, const std::string& in_module_fullname,
            const std::string& in_search, megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state);

    private:
        typedef std::vector<megamol::gui::Parameter*> ParamPtrVector_t;
        typedef std::map<std::string, ParamPtrVector_t> ParamGroup_t;

        // VARIABLES --------------------------------------------------------------

        HoverToolTip tooltip;
        ParameterGroupViewCubeWidget cube_widget_group;
        ParameterGroupAnimationWidget animation_group;

        std::vector<AbstractParameterGroupWidget*> group_widgets;
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED

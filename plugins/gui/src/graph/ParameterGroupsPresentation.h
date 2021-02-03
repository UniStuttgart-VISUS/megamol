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
#include "widgets/WidgetPicking_gl.h"

#include "mmcore/param/AbstractParamPresentation.h"

#include "vislib/math/Ternary.h"


namespace megamol {
namespace gui {


    /** ************************************************************************
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

    private:
        class GroupWidgetData;

        typedef std::map<std::string, GroupWidgetData> GroupWidgets_t;
        typedef std::pair<const std::string, GroupWidgetData> GroupWidgetData_t;

        typedef std::vector<megamol::gui::Parameter*> ParamPtrVector_t;
        typedef std::map<std::string, ParamPtrVector_t> ParamGroup_t;

        typedef std::function<bool(bool only_check, ParamPtrVector_t& params)> GroupWidgetCheckCallbackFunc_t;

        typedef std::function<bool(GroupWidgetData_t& group_widget_data, ParamPtrVector_t params,
            const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer)>
            GroupWidgetDrawCallbackFunc_t;

        // Data needed for group widgets
        class GroupWidgetData : public megamol::core::param::AbstractParamPresentation {
        public:
            GroupWidgetData(void) : megamol::core::param::AbstractParamPresentation(){};
            GroupWidgetData(Param_t pt) : megamol::core::param::AbstractParamPresentation() {
                this->InitPresentation(pt);
            }
            ~GroupWidgetData(void) = default;

            bool active;
            GroupWidgetCheckCallbackFunc_t check_callback;
            GroupWidgetDrawCallbackFunc_t draw_callback;
        };


        // VARIABLES --------------------------------------------------------------

        GroupWidgets_t group_widgets;

        // Widgets
        HoverToolTip tooltip;
        PickableCube cube_widget;

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

        void draw_grouped_parameters(const std::string& in_group_name, ParamPtrVector_t& params,
            const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state);

        bool check_group_widget_animation(bool only_check, ParamPtrVector_t& params);
        bool draw_group_widget_animation(GroupWidgetData_t& group_widget_data, ParamPtrVector_t params,
            const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer);

        bool check_group_widget_3d_cube(bool only_check, ParamPtrVector_t& params);
        bool draw_group_widget_3d_cube(GroupWidgetData_t& group_widget_data, ParamPtrVector_t params,
            const std::string& in_module_fullname, const std::string& in_search,
            megamol::gui::ParameterPresentation::WidgetScope in_scope,
            const std::shared_ptr<TransferFunctionEditor> in_external_tf_editor, bool* out_open_external_tf_editor,
            ImGuiID in_override_header_state, PickingBuffer* inout_picking_buffer);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED

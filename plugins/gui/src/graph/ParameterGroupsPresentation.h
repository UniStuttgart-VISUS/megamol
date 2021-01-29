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
        typedef std::vector<megamol::gui::Parameter*> ParamPtrVector_t;
        typedef std::map<std::string, ParamPtrVector_t> ParamGroup_t;
        typedef std::function<bool(ParamPtrVector_t& params, const std::string& in_search,
            megamol::core::param::AbstractParamPresentation::Presentation presentation,
            megamol::gui::ParameterPresentation::WidgetScope in_scope, PickingBuffer* inout_picking_buffer)>
            GroupWidgetCallbackFunc_t;


        // Data needed for group widgets
        class GroupWidgetData : public megamol::core::param::AbstractParamPresentation {
        public:
            GroupWidgetData(void) : megamol::core::param::AbstractParamPresentation(){};
            GroupWidgetData(Param_t pt) : megamol::core::param::AbstractParamPresentation() {
                this->InitPresentation(pt);
            }
            ~GroupWidgetData(void) = default;
            bool active;
            GroupWidgetCallbackFunc_t callback;
        };

        // VARIABLES --------------------------------------------------------------

        std::map<std::string, GroupWidgetData> group_widget_ids;

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

        bool group_widget_animation(ParamPtrVector_t& params, const std::string& in_search,
            megamol::core::param::AbstractParamPresentation::Presentation presentation,
            megamol::gui::ParameterPresentation::WidgetScope in_scope);

        bool group_widget_3d_cube(ParamPtrVector_t& params, const std::string& in_search,
            megamol::core::param::AbstractParamPresentation::Presentation presentation,
            megamol::gui::ParameterPresentation::WidgetScope in_scope, PickingBuffer* inout_picking_buffer);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_PARAMETERGROUPS_H_INCLUDED

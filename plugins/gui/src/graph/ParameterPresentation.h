/*
 * Parameter.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_GRAPH_PARAMETER_PRESENTATION_H_INCLUDED
#define MEGAMOL_GUI_GRAPH_PARAMETER_PRESENTATION_H_INCLUDED


#include "FileUtils.h"
#include "GUIUtils.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/ImageWidget_gl.h"
#include "widgets/ParameterOrbitalWidget.h"
#include "widgets/TransferFunctionEditor.h"

#include "mmcore/param/BoolParam.h"
#include "mmcore/param/ButtonParam.h"
#include "mmcore/param/ColorParam.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FilePathParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/TernaryParam.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"

#include <variant>

// Used for platform independent clipboard (ImGui so far only provides windows implementation)
#ifdef GUI_USE_GLFW
#    include "GLFW/glfw3.h"
#endif


namespace megamol {
namespace gui {


// Forward declarations
class Parameter;


/** ************************************************************************
 * Defines GUI parameter presentation.
 */
class ParameterPresentation : public megamol::core::param::AbstractParamPresentation {
public:
    friend class Parameter;

    /*
     * Globally scoped widgets (widget parts) are always called each frame.
     * Locally scoped widgets (widget parts) are only called if respective parameter appears in GUI.
     */
    enum WidgetScope { GLOBAL, LOCAL };

    // VARIABLES --------------------------------------------------------------

    bool extended;
    const std::string float_format;

    // FUCNTIONS --------------------------------------------------------------

    ParameterPresentation(Param_t type);
    ~ParameterPresentation(void);

    bool IsGUIStateDirty(void) { return this->guistate_dirty; }
    void ResetGUIStateDirty(void) { this->guistate_dirty = false; }
    void ForceSetGUIStateDirty(void) { this->guistate_dirty = true; }

    void SetTransferFunctionEditorHash(size_t hash) { this->tf_editor_hash = hash; }

    inline void ConnectExternalTransferFunctionEditor(std::shared_ptr<megamol::gui::TransferFunctionEditor> tfe_ptr) {
        if (this->tf_editor_external_ptr != tfe_ptr) {
            this->tf_editor_external_ptr = tfe_ptr;
            this->use_external_tf_editor = true;
        }
    }

    void LoadTransferFunctionTexture(
        std::vector<float>& in_texture_data, int& in_texture_width, int& in_texture_height);

    /** "Point in Circle" Button for additional drop down Options. */
    static bool OptionButton(const std::string& id, const std::string& label = "", bool dirty = false);

    /** Knob button for 'circular' float value manipulation. */
    static bool KnobButton(const std::string& id, float size, float& inout_value, float minval, float maxval);

    /** Extended parameter mode button. */
    static bool ParameterExtendedModeButton(bool& inout_extended_mode);

private:
    // VARIABLES --------------------------------------------------------------

    std::string help;
    std::string description;
    std::variant<std::monostate, std::string, int, float, glm::vec2, glm::vec3, glm::vec4> widget_store;
    unsigned int set_focus;
    bool guistate_dirty;

    std::shared_ptr<megamol::gui::TransferFunctionEditor> tf_editor_external_ptr;
    megamol::gui::TransferFunctionEditor tf_editor_internal;
    bool use_external_tf_editor;
    bool show_tf_editor;
    size_t tf_editor_hash;

    // Widgets
    FileBrowserWidget file_browser;
    HoverToolTip tooltip;
    ImageWidget image_widget;
    ParameterOrbitalWidget rotation_widget;

    // FUNCTIONS --------------------------------------------------------------
    bool Present(Parameter& inout_param, WidgetScope scope);

    bool present_parameter(Parameter& inout_parameter, WidgetScope scope);

    bool widget_button(WidgetScope scope, const std::string& label, const megamol::core::view::KeyCode& keycode);
    bool widget_bool(WidgetScope scope, const std::string& label, bool& value);
    bool widget_string(WidgetScope scope, const std::string& label, std::string& value);
    bool widget_color(WidgetScope scope, const std::string& label, glm::vec4& value);
    bool widget_enum(WidgetScope scope, const std::string& label, int& value, EnumStorage_t storage);
    bool widget_flexenum(WidgetScope scope, const std::string& label, std::string& value,
        megamol::core::param::FlexEnumParam::Storage_t storage);
    bool widget_filepath(WidgetScope scope, const std::string& label, std::string& value);
    bool widget_ternary(WidgetScope scope, const std::string& label, vislib::math::Ternary& value);
    bool widget_int(WidgetScope scope, const std::string& label, int& value, int minval, int maxval);
    bool widget_float(WidgetScope scope, const std::string& label, float& value, float minval, float maxval);
    bool widget_vector2f(
        WidgetScope scope, const std::string& label, glm::vec2& value, glm::vec2 minval, glm::vec2 maxval);
    bool widget_vector3f(
        WidgetScope scope, const std::string& label, glm::vec3& value, glm::vec3 minval, glm::vec3 maxval);
    bool widget_vector4f(
        WidgetScope scope, const std::string& label, glm::vec4& value, glm::vec4 minval, glm::vec4 maxval);
    bool widget_pinvaluetomouse(WidgetScope scope, const std::string& label, const std::string& value);
    bool widget_transfer_function_editor(WidgetScope scope, Parameter& inout_parameter);
    bool widget_knob(WidgetScope scope, const std::string& label, float& value, float minval, float maxval);
    bool widget_rotation_axes(
        WidgetScope scope, const std::string& label, glm::vec4& value, glm::vec4 minval, glm::vec4 maxval);
    bool widget_rotation_direction(
        WidgetScope scope, const std::string& label, glm::vec3& value, glm::vec3 minval, glm::vec3 maxval);
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GRAPH_PARAMETER_PRESENTATION_H_INCLUDED

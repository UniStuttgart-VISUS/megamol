/*
 * AnimationEditor.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractWindow.h"
#include "CommonTypes.h"
#include "animation/AnimationData.h"
#include "imgui_canvas.h"
#include "mmcore/MegaMolGraph.h"
#include "widgets/FileBrowserWidget.h"

#include <variant>

namespace megamol {
namespace gui {

enum class InteractionType : int32_t {
    None = 0,
    DraggingKey,
    DraggingLeftTangent,
    DraggingRightTangent,
    DraggingPlayhead
};

class AnimationEditor : public AbstractWindow {
public:
    using lua_func_type = megamol::frontend_resources::common_types::lua_func_type;

    explicit AnimationEditor(const std::string& window_name);
    ~AnimationEditor();

    bool Update() override;
    bool Draw() override;

    void SetLuaFunc(lua_func_type* func);

    bool NotifyParamChanged(
        frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param, std::string const& new_value);

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

private:
    using animations = std::variant<animation::FloatAnimation, animation::StringAnimation>;

    void WriteValuesToGraph();
    bool SaveToFile(const std::string& file);
    void ClearData();
    bool LoadFromFile(std::string file);

    void DrawPopups();
    void DrawToolbar();
    void CenterAnimation(const animations& anim);
    void SelectAnimation(int32_t a);
    void DrawParams();
    void DrawInterpolation(ImDrawList* dl, const animation::FloatKey& key, const animation::FloatKey& k2);
    void DrawFloatKey(ImDrawList* dl, animation::FloatKey& key);
    void DrawPlayhead(ImDrawList* drawList);
    void DrawStringKey(ImDrawList* im_draws, animation::StringKey& key);
    void DrawCurves();
    void DrawProperties();

    void DrawVerticalSeparator();

    void DrawGrid(const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment,
        float sign = 1.0f);
    void DrawScale(const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment,
        float sign = 1.0f);
    // VARIABLES --------------------------------------------------------------

    std::vector<animations> allAnimations;
    //std::vector<animation::FloatAnimation> floatAnimations;
    //std::vector<animation::StringAnimation> stringAnimations;
    int32_t selectedAnimation = -1;
    animation::FloatKey* selectedFloatKey = nullptr;
    animation::FloatKey* draggingFloatKey = nullptr;
    animation::StringKey* selectedStringKey = nullptr;
    animation::StringKey* draggingStreingKey = nullptr;
    ImGuiEx::Canvas canvas = ImGuiEx::Canvas();

    bool canvas_visible = false;
    bool is_dragging = false;
    bool playback_active = false;
    ImVec2 drag_start = {0.0f, 0.0f};
    float drag_start_value = 0.0f;
    animation::KeyTimeType drag_start_time = 0;
    bool auto_capture = false;
    bool write_to_graph = false;
    InteractionType curr_interaction = InteractionType::None;

    animation::KeyTimeType current_frame = 0;
    animation::KeyTimeType animation_bounds[2] = {0, 100};

    ImVec2 custom_zoom = {1.0f, 1.0f};

    lua_func_type* input_lua_func = nullptr;
    FileBrowserWidget file_browser;
    bool open_popup_load = false, open_popup_save = false, open_popup_error = false;
    std::string error_popup_message;
    vislib::math::Ternary ternary = vislib::math::Ternary::TRI_FALSE;
    std::string animation_file;
};


} // namespace gui
} // namespace megamol

/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once


#include "AbstractWindow.h"
#include "AnimationEditorData.h"
#include "CommonTypes.h"
#include "imgui_canvas.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/utility/animation/AnimationData.h"
#include "widgets/FileBrowserWidget.h"

#include <variant>

namespace megamol::gui {

using namespace core::utility;

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
    void SetLastFrameMillis(float last_frame_ms);

    void RenderAnimation();
    std::string GenerateLuaForFrame(animation::KeyTimeType time);

    frontend_resources::AnimationEditorData& GetAnimationEditorData() {
        return this->animEditorData;
    }

private:
    using animations =
        std::variant<animation::FloatAnimation, animation::StringAnimation, animation::FloatVectorAnimation>;

    void WriteValuesToGraph();
    bool SaveToFile(const std::string& file);
    void ClearData();
    bool LoadFromFile(std::string file);

    bool AnyParamHasKey(animation::KeyTimeType key_time);
    bool ExportToFile(const std::string& export_file);
    void DrawPopups();
    void DrawToolbar();
    void CenterAnimation(const animations& anim);
    void SelectAnimation(int32_t a);
    void DrawParams();
    void DrawInterpolation(ImDrawList* dl, const animation::FloatKey& key, const animation::FloatKey& k2,
        ImU32 col = ImGui::GetColorU32(ImGuiCol_NavHighlight));
    void DrawFloatKey(ImDrawList* dl, animation::FloatKey& key, ImU32 col = IM_COL32(255, 128, 0, 255),
        animation::VectorKey<animation::FloatKey>* parent = nullptr);
    void DrawPlayhead(ImDrawList* drawList);
    void DrawStringKey(ImDrawList* im_draws, animation::StringKey& key, ImU32 col = IM_COL32(255, 128, 0, 255));
    void DrawCurves();
    void DrawProperties();

    void DrawVerticalSeparator();

    void DrawGrid();
    void DrawScale();

    // VARIABLES --------------------------------------------------------------

    std::vector<animations> allAnimations;
    int32_t selectedAnimation = -1;
    animation::FloatKey* selectedFloatKey = nullptr;
    animation::FloatKey* draggingFloatKey = nullptr;
    animation::StringKey* selectedStringKey = nullptr;
    animation::VectorKey<animation::FloatKey>* current_parent = nullptr;
    ImGuiEx::Canvas canvas = ImGuiEx::Canvas();

    bool canvas_visible = false;
    bool is_dragging = false;
    ImVec2 drag_start = {0.0f, 0.0f};
    float drag_start_value = 0.0f;
    animation::KeyTimeType drag_start_time = 0;
    bool auto_capture = false;
    bool write_to_graph = false;
    InteractionType curr_interaction = InteractionType::None;

    animation::KeyTimeType current_frame = 0;
    int32_t playback_fps = 30;
    float targeted_frame_time = 1000.0f / static_cast<float>(playback_fps);
    float last_frame_ms = 0.0f, accumulated_ms = 0.0f;

    int32_t playing = 0;
    bool rendering = false;
    animation::KeyTimeType animation_bounds[2] = {0, 100};

    ImVec2 custom_zoom = {1.0f, 1.0f};

    lua_func_type* input_lua_func = nullptr;
    FileBrowserWidget file_browser;
    bool open_popup_load = false, open_popup_save = false, open_popup_error = false;
    bool open_popup_export = false;
    int export_screenshot_option = 0;
    bool propagate_to_siblings = false;
    std::string error_popup_message;
    bool save_state = false, save_all_params = false;
    std::string animation_file, export_file;
    std::string output_prefix;

    frontend_resources::AnimationEditorData animEditorData;
};


} // namespace megamol::gui

/*
 * AnimationEditor.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractWindow.h"
#include "CommonTypes.h"
#include "WindowCollection.h"
#include "mmcore/MegaMolGraph.h"
#include "imgui_canvas.h"

#include <map>


namespace megamol {
namespace gui {

using KeyTimeType = int32_t;

enum class InterpolationType : int32_t {
    Step = 0,
    Linear = 1,
    Hermite = 2,
    CubicBezier = 3
};

enum class InteractionType : int32_t {
    None = 0,
    DraggingKey,
    DraggingLeftTangent,
    DraggingRightTangent,
    DraggingPlayhead
};

struct Key {
    KeyTimeType time;
    float value;
    InterpolationType interpolation = InterpolationType::Linear;
    bool tangents_linked = true;
    ImVec2 in_tangent{-1.0f, 0.0f};
    ImVec2 out_tangent{1.0f, 0.0f};

    // this is expensive (accurately hit time first!)...
    static float Interpolate(Key first, Key second, KeyTimeType time);
    // ... and that is only good for drawing (x will not sit on the time grid)
    static ImVec2 Interpolate(Key first, Key second, float t);
};

class FloatAnimation {
public:
    using KeyMap = std::map<KeyTimeType, Key>;
    using ValueType = float;

    FloatAnimation(std::string ParamName) : param_name(ParamName) {}
    void AddKey(Key k);
    void DeleteKey(KeyTimeType time);

    KeyMap::iterator begin();
    KeyMap::iterator end();

    ValueType GetValue(KeyTimeType time) const;
    const std::string& GetName() const;
    KeyTimeType GetStartTime() const;
    KeyTimeType GetEndTime() const;
    KeyTimeType GetLength() const;
    ValueType GetMinValue() const;
    ValueType GetMaxValue() const;
    KeyMap::size_type GetSize() const {
        return keys.size();
    }
    bool HasKey(KeyTimeType k) {
        return keys.find(k) != keys.end();
    }
    Key& operator[](KeyTimeType k) {
        return keys[k];
    }
    std::vector<KeyTimeType> GetAllKeys() const {
        std::vector<KeyTimeType> the_keys;;
        the_keys.reserve(keys.size());
        std::transform(keys.begin(), keys.end(), std::back_inserter(the_keys),
            [](const KeyMap::value_type& v) { return v.first; });
        return the_keys;
    }

private:
    KeyMap keys;
    std::string param_name;
    KeyTimeType timeline_offset = 0;
    KeyTimeType playhead_position = 0;
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
    void WriteValuesToGraph();

    void DrawToolbar();
    void center_animation(const FloatAnimation& anim);
    void DrawParams();
    void DrawInterpolation(ImDrawList* dl, const Key& key, const Key& k2);
    void DrawKey(ImDrawList* dl, Key& key);
    void DrawPlayhead(ImDrawList* drawList);
    void DrawCurves();
    void DrawProperties();

    void DrawVerticalSeparator();

    void DrawGrid(
        const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment, float sign = 1.0f);
    void DrawScale(const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment,
        float sign = 1.0f);
    // VARIABLES --------------------------------------------------------------

    std::vector<FloatAnimation> floatAnimations;
    int32_t selectedAnimation = -1;
    Key* selectedKey = nullptr;
    Key* draggingKey = nullptr;
    ImGuiEx::Canvas canvas = ImGuiEx::Canvas();

    bool canvas_visible = false;
    bool is_dragging = false;
    bool playback_active = false;
    ImVec2 drag_start = {0.0f, 0.0f};
    float drag_start_value = 0.0f;
    KeyTimeType drag_start_time = 0;
    bool auto_capture = false;
    bool write_to_graph = false;
    InteractionType curr_interaction = InteractionType::None;

    KeyTimeType current_frame = 0;
    KeyTimeType animation_bounds[2] = {0, 100};

    ImVec2 custom_zoom = {1.0f, 1.0f};

    lua_func_type* input_lua_func;
};


} // namespace gui
} // namespace megamol

/*
 * AnimationEditor.h
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractWindow.h"
#include "WindowCollection.h"
#include "mmcore/MegaMolGraph.h"
#include "widgets/imgui_canvas.h"

#include <map>


namespace megamol {
namespace gui {

using KeyTimeType = int32_t;

enum class InterpolationType : int32_t {
    Step = 0,
    Linear = 1,
    Hermite = 2
};

struct Key {
    KeyTimeType time;
    float value;
    InterpolationType interpolation = InterpolationType::Linear;
    bool tangents_linked = true;
    ImVec2 in_tangent{-10.0f, 0.0f};
    ImVec2 out_tangent{10.0f, 0.0f};

    static float Interpolate(Key first, Key second, KeyTimeType time);
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
    ValueType GetMinValue() const;
    ValueType GetMaxValue() const;
    KeyMap::size_type GetSize() const {
        return keys.size();
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
    explicit AnimationEditor(const std::string& window_name);
    ~AnimationEditor();

    bool Update() override;
    bool Draw() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;


private:
    void DrawToolbar();
    void DrawParams();
    void DrawInterpolation(ImDrawList* dl, const Key& key, const Key& k2);
    void DrawKey(ImDrawList* dl, Key& key);
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
    ImGuiEx::Canvas canvas = ImGuiEx::Canvas();
    KeyTimeType anim_start = 0;
    KeyTimeType anim_end = 100;

    const float frame_width = 10.0f;
    const float value_scale = 10.0f;
    bool canvas_visible = false;
    bool is_dragging = false;
    ImVec2 drag_start = {0.0f, 0.0f};
    float drag_start_value = 0.0f;
    KeyTimeType drag_start_time = 0;
    float zoom = 1.0f;
    bool auto_capture = false;

    ImVec2 custom_zoom = {1.0f, 1.0f};
};


} // namespace gui
} // namespace megamol

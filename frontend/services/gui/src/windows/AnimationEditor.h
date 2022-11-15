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

using KeyTimeType = uint32_t;

enum class InterpolationType {
    Step,
    Linear,
    Hermite
};

struct Tangent {
    float length;
    float offset;
};

struct Key {
    KeyTimeType time;
    float value;
    InterpolationType interpolation;
    Tangent in_tangent;
    Tangent out_tangent;

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
    void DrawCurves();
    void DrawProperties();

    void DrawVerticalSeparator();

    void DrawGrid(
        const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment, float sign = 1.0f);
    static void DrawScale(const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment,
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
    float zoom = 1.0f;

};


} // namespace gui
} // namespace megamol

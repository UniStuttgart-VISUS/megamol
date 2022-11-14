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
    FloatAnimation(std::string ParamName) : param_name(ParamName) {}
    void AddKey(Key k);
    void DeleteKey(KeyTimeType time);

    float GetValue(KeyTimeType time);

private:
    std::map<KeyTimeType, Key> keys;
    std::string param_name;
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
    // VARIABLES --------------------------------------------------------------

};


} // namespace gui
} // namespace megamol

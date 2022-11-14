/*
 * AnimationEditor.cpp
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "AnimationEditor.h"


using namespace megamol::gui;


float Key::Interpolate(Key first, Key second, KeyTimeType time) {
    Key my_first = first;
    Key my_second = second;
    if (my_first.time > my_second.time) {
        my_first = second;
        my_second = first;
    }
    if (time <= my_first.time) {
        return my_first.value;
    }
    if (time >= my_second.time) {
        return my_second.value;
    }

    const float t = static_cast<float>(time - my_first.time) / static_cast<float>(my_second.time - my_first.time);

    switch (first.interpolation) {
    case InterpolationType::Step:
        return my_first.value;
    case InterpolationType::Linear:
        return (1.0f - t) * my_first.value + t * my_second.value;
    case InterpolationType::Hermite:
        const auto t2 = t * t, t3 = t2 * t;
        return my_first.value * (2.0f * t3 - 3.0f * t2 + 1.0f) + my_second.value * (-2.0f * t3 + 3.0f * t2) +
               my_first.out_tangent.length * my_first.out_tangent.offset * (t3 - 2.0f * t2 + t) +
               my_second.in_tangent.length * my_second.in_tangent.offset * (t3 - t2);
    }
    return 0.0f;
}


void FloatAnimation::AddKey(Key k) {
    keys[k.time] = k;
}


void FloatAnimation::DeleteKey(KeyTimeType time) {
    keys.erase(time);
}

float FloatAnimation::GetValue(KeyTimeType time) {
    if (keys.size() < 2) return 0.0f;
    Key before_key = keys[0], after_key = keys[0];
    bool ok = false;
    for (auto it = keys.begin(); it != keys.end(); ++it) {
        if (it->second.time < time) {
            before_key = it->second;
        }
        if (it->second.time > time) {
            after_key = it->second;
            ok = true;
            break;
        }
    }
    if (ok) {
        return Key::Interpolate(before_key, after_key, time);
    } else {
        return 0.0f;
    }
}


megamol::gui::AnimationEditor::AnimationEditor(const std::string& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_ANIMATIONEDITOR) {

    // Configure HOTKEY EDITOR Window
    this->win_config.size = ImVec2(0.0f * megamol::gui::gui_scaling.Get(), 0.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_NoNavInputs;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F5, core::view::Modifier::NONE);

    // TODO we actually want to subscribe to graph updates to be able to "record" parameter changes automatically
    // TODO what do we do with parameters other than floats?
}


megamol::gui::AnimationEditor::~AnimationEditor() {

}


bool megamol::gui::AnimationEditor::Update() {

    return true;
}


bool megamol::gui::AnimationEditor::Draw() {
    // Toolbar

    // Tree/List of available Params
    // Curve Editor
    // Property Window for Selection
    return true;
}


void megamol::gui::AnimationEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {

}


void megamol::gui::AnimationEditor::SpecificStateToJSON(nlohmann::json& inout_json) {

}


/*
 * AnimationEditor.cpp
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "AnimationEditor.h"

#include "animation/AnimationUtils.h"
#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"

#include "imgui_stdlib.h"
#include "nlohmann/json.hpp"

#include <fstream>

using namespace megamol::gui;

AnimationEditor::AnimationEditor(const std::string& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_ANIMATIONEDITOR) {

    // Configure ANIMATION EDITOR Window
    this->win_config.size = ImVec2(1600.0f * megamol::gui::gui_scaling.Get(), 800.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_NoNavInputs | ImGuiWindowFlags_MenuBar;
    this->win_config.hotkey = core::view::KeyCode(core::view::Key::KEY_F5, core::view::Modifier::NONE);
}


AnimationEditor::~AnimationEditor() {}


bool AnimationEditor::Update() {

    return true;
}


bool AnimationEditor::Draw() {
    if (playing != 0) {
        accumulated_ms += last_frame_ms;
        if (accumulated_ms > targeted_frame_time) {
            auto frameskip = static_cast<int>(accumulated_ms / targeted_frame_time);
            accumulated_ms = std::fmodf(accumulated_ms, targeted_frame_time);
            current_frame += frameskip * playing;
            if (playing > 0) {
                if (current_frame > animation_bounds[1]) {
                    current_frame -= animation_bounds[1] + 1;
                    current_frame += animation_bounds[0];
                }
            } else {
                if (current_frame < animation_bounds[0]) {
                    current_frame += animation_bounds[1] + 1;
                    current_frame -= animation_bounds[0];
                }
            }
        }
    }
    DrawToolbar();
    DrawPopups();

    ImGui::BeginChild("AnimEditorContent");
    ImGui::Columns(3, "AnimEditorColumns", false);
    ImGui::SetColumnWidth(0, ImGui::GetWindowSize().x / 5.0f);
    ImGui::SetColumnWidth(1, ImGui::GetWindowSize().x * (3.0f / 5.0f));
    ImGui::SetColumnWidth(2, ImGui::GetWindowSize().x / 5.0f);
    DrawParams();
    ImGui::NextColumn();
    DrawCurves();
    ImGui::NextColumn();
    DrawProperties();
    ImGui::EndChild();

    return true;
}


void AnimationEditor::SetLuaFunc(lua_func_type* func) {
    this->input_lua_func = func;
}


bool AnimationEditor::NotifyParamChanged(
    frontend_resources::ModuleGraphSubscription::ParamSlotPtr const& param_slot, std::string const& new_value) {
    if (auto_capture) {
        std::string the_name = param_slot->FullName().PeekBuffer();
        int found_idx = -1;
        for (auto i = 0; i < allAnimations.size(); ++i) {
            auto& anim = allAnimations[i];
            bool found = std::visit(
                [&](auto&& arg) -> bool {
                    if (arg.GetName() == the_name) {
                        found_idx = i;
                        return true;
                    } else {
                        return false;
                    }
                },
                anim);
            if (found) {
                break;
            }
        }

        if (param_slot->Param<FloatParam>() || param_slot->Param<IntParam>()) {
            const float the_val = std::stof(new_value);
            if (found_idx != -1) {
                auto& a = std::get<animation::FloatAnimation>(allAnimations[found_idx]);
                if (a.HasKey(current_frame)) {
                    a[current_frame].value = the_val;
                } else {
                    a.AddKey({current_frame, the_val});
                }
                CenterAnimation(a);
            }
            if (found_idx == -1) {
                animation::FloatAnimation f = {the_name};
                f.AddKey({current_frame, the_val});
                allAnimations.emplace_back(f);
                CenterAnimation(f);
            }
            selectedStringKey = nullptr;
        }
        if (param_slot->Param<EnumParam>() || param_slot->Param<FlexEnumParam>() || param_slot->Param<StringParam>() ||
            param_slot->Param<FilePathParam>()) {
            if (found_idx != -1) {
                auto& a = std::get<animation::StringAnimation>(allAnimations[found_idx]);
                if (a.HasKey(current_frame)) {
                    a[current_frame].value = new_value;
                } else {
                    a.AddKey({current_frame, new_value});
                }
                CenterAnimation(a);
            }
            if (found_idx == -1) {
                animation::StringAnimation s = {the_name};
                s.AddKey({current_frame, new_value});
                allAnimations.emplace_back(s);
                CenterAnimation(s);
            }
            selectedFloatKey = nullptr;
        }
    }
    // TODO: what else. Enum probably.
    return true;
}


void AnimationEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {
    in_json["write_to_graph"].get_to(write_to_graph);
    in_json["playing"].get_to(playing);
    in_json["playback_fps"].get_to(playback_fps);
    in_json["animation_bounds"].get_to(animation_bounds);
    in_json["current_frame"].get_to(current_frame);
    if (in_json.contains("animation_file")) {
        in_json["animation_file"].get_to(animation_file);
    }

    animation::FloatAnimation f_dummy("dummy");
    animation::StringAnimation s_dummy("dummy");
    if (!in_json.contains("animations")) {
        error_popup_message = "Loading failed: cannot find 'animations'";
        open_popup_error = true;
    }
    for (auto& j : in_json["animations"]) {
        if (j["type"] == "string") {
            from_json(j, s_dummy);
            allAnimations.emplace_back(s_dummy);
        } else if (j["type"] == "float") {
            // j.get<FloatAnimation>() does not work because no default constructor
            from_json(j, f_dummy);
            allAnimations.emplace_back(f_dummy);
        } else {
            error_popup_message = "Loading failed: " + j["name"].get<std::string>() + " has no type";
            open_popup_error = true;
        }
    }
}


void AnimationEditor::SpecificStateToJSON(nlohmann::json& inout_json) {
    inout_json["write_to_graph"] = write_to_graph;
    inout_json["playing"] = playing;
    inout_json["playback_fps"] = playback_fps;
    inout_json["animation_bounds"] = animation_bounds;
    inout_json["current_frame"] = current_frame;
    inout_json["animation_file"] = animation_file;

    nlohmann::json anims;
    for (auto& fa : allAnimations) {
        std::visit([&](auto&& arg) -> void { anims.push_back(arg); }, fa);
    }
    inout_json["animations"] = anims;
}


void AnimationEditor::SetLastFrameMillis(float last_frame_millis) {
    last_frame_ms = last_frame_millis;
}


void AnimationEditor::WriteValuesToGraph() {
    if (write_to_graph) {
        std::stringstream lua_commands;
        for (auto a : allAnimations) {
            std::visit(
                [&](auto&& arg) -> void {
                    lua_commands << "mmSetParamValue(\"" << arg.GetName() << "\",[=[" << arg.GetValue(current_frame)
                                 << "]=])\n";
                },
                a);
        }
        auto res = (*input_lua_func)(lua_commands.str());
        if (!std::get<0>(res)) {
            open_popup_error = true;
            error_popup_message = std::get<1>(res);
            playing = 0;
        }
    }
}


bool AnimationEditor::SaveToFile(const std::string& file) {
    std::ofstream out(file);
    if (!out.is_open()) {
        return false;
    }
    nlohmann::json animation_data;
    SpecificStateToJSON(animation_data);
    animation_data.erase("animation_file");
    auto s = animation_data.dump(2);
    out << s;
    out.close();
    return true;
}


void AnimationEditor::ClearData() {
    allAnimations.clear();
    selectedAnimation = -1;
    selectedFloatKey = nullptr;
    animation_file = "";
}

bool AnimationEditor::LoadFromFile(std::string file) {
    ClearData();
    std::ifstream in(file);
    if (!in.is_open()) {
        return false;
    }
    auto j_all = nlohmann::json::parse(in);
    SpecificStateFromJSON(j_all);
    return true;
}


void AnimationEditor::DrawPopups() {
    if (this->file_browser.PopUp_Load(
            "Load Animation", animation_file, open_popup_load, {"anim"}, FilePathParam::Flag_File_RestrictExtension)) {
        LoadFromFile(animation_file);
        open_popup_load = false;
    }
    if (this->file_browser.PopUp_Save("Save Animation", animation_file, open_popup_save, {"anim"},
            FilePathParam::Flag_File_ToBeCreatedWithRestrExts, ternary)) {
        if (!SaveToFile(animation_file)) {
            error_popup_message = "Saving failed.";
            open_popup_error = true;
        }
        open_popup_save = false;
    }

    if (open_popup_error) {
        const ImVec2 center = ImGui::GetMainViewport()->GetCenter();
        ImGui::SetNextWindowPos(center, 0, ImVec2(0.5f, 0.5f));
        ImGui::OpenPopup("AnimEditorError");
    }
    if (ImGui::BeginPopupModal("AnimEditorError", &open_popup_error,
            ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoScrollbar)) {
        ImGui::Text("error: %s", error_popup_message.c_str());
        if (ImGui::Button("OK", ImVec2(120, 0))) {
            ImGui::CloseCurrentPopup();
            open_popup_error = false;
        }
        ImGui::EndPopup();
    }
}

void AnimationEditor::DrawToolbar() {
    // Menu
    ImGui::PushID("AnimationEditor::Menu");
    if (ImGui::BeginMenuBar()) {
        if (ImGui::BeginMenu("Animation")) {
            if (ImGui::MenuItem("Clear all")) {
                ClearData();
            }
            if (ImGui::MenuItem("Load...")) {
                open_popup_load = true;
            }
            if (animation_file.empty()) {
                ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
            }
            if (ImGui::MenuItem("Save")) {
                if (!SaveToFile(animation_file)) {
                    error_popup_message = "Saving failed.";
                    open_popup_error = true;
                }
            }
            if (animation_file.empty()) {
                ImGui::PopItemFlag();
            }
            if (ImGui::MenuItem("Save as...")) {
                open_popup_save = true;
            }
            ImGui::EndMenu();
        }
        ImGui::EndMenuBar();
    }
    ImGui::PopID();

    // what will be icons some day?
    // graph connections
    if (ImGui::Checkbox("auto capture", &auto_capture)) {
        if (auto_capture) {
            write_to_graph = false;
        }
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("write to graph", &write_to_graph)) {
        if (write_to_graph) {
            auto_capture = false;
        }
    }
    ImGui::SameLine();
    DrawVerticalSeparator();

    // keys
    ImGui::SameLine();
    if (ImGui::Button("add")) {
        if (selectedAnimation != -1) {
            if (std::holds_alternative<animation::FloatAnimation>(allAnimations[selectedAnimation])) {
                auto& anim = std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]);
                if (!anim.HasKey(current_frame)) {
                    animation::FloatKey f;
                    f.time = current_frame;
                    f.value = anim.GetValue(current_frame);
                    f.interpolation = anim.GetInterpolation(current_frame);
                    anim.AddKey(f);
                }
            }
            if (std::holds_alternative<animation::StringAnimation>(allAnimations[selectedAnimation])) {
                auto& anim = std::get<animation::StringAnimation>(allAnimations[selectedAnimation]);
                if (!anim.HasKey(current_frame)) {
                    animation::StringKey s;
                    s.time = current_frame;
                    s.value = std::string(anim.GetValue(current_frame));
                    anim.AddKey(s);
                }
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("delete")) {
        if (selectedAnimation != -1) {
            if (std::holds_alternative<animation::FloatAnimation>(allAnimations[selectedAnimation]) &&
                selectedFloatKey != nullptr) {
                std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]).DeleteKey(selectedFloatKey->time);
                selectedFloatKey = nullptr;
            }
            if (std::holds_alternative<animation::StringAnimation>(allAnimations[selectedAnimation]) &&
                selectedStringKey != nullptr) {
                std::get<animation::StringAnimation>(allAnimations[selectedAnimation])
                    .DeleteKey(selectedStringKey->time);
                selectedStringKey = nullptr;
            }
        }
    }
    ImGui::SameLine();
    DrawVerticalSeparator();

    // tangent controls
    ImGui::SameLine();
    if (ImGui::Button("break tangents")) {
        if (selectedFloatKey != nullptr) {
            selectedFloatKey->tangents_linked = false;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("link tangents")) {
        if (selectedFloatKey != nullptr) {
            selectedFloatKey->tangents_linked = true;
            selectedFloatKey->out_tangent = ImVec2(-selectedFloatKey->in_tangent.x, -selectedFloatKey->in_tangent.y);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("flat tangents")) {
        if (selectedFloatKey != nullptr) {
            selectedFloatKey->in_tangent = {-1.0f, 0.0f};
            selectedFloatKey->out_tangent = {1.0f, 0.0f};
        }
    }
    ImGui::SameLine();
    DrawVerticalSeparator();

    // view controls
    ImGui::SameLine();
    if (ImGui::Button("frame view") && selectedAnimation != -1) {
        CenterAnimation(allAnimations[selectedAnimation]);
    }
    ImGui::SameLine();
    ImGui::PushItemWidth(100.0f);
    ImGui::SliderFloat("HZoom", &custom_zoom.x, 0.01f, 5000.0f);
    ImGui::SameLine();
    ImGui::SliderFloat("VZoom", &custom_zoom.y, 0.01f, 5000.0f);
}


void AnimationEditor::CenterAnimation(const animations& anim) {
    auto region = canvas.Rect();
    float min_val = -1.0f, max_val = 1.0f, start = 0, len = 1;
    if (std::holds_alternative<animation::FloatAnimation>(anim)) {
        auto& a = std::get<animation::FloatAnimation>(anim);
        min_val = a.GetMinValue();
        max_val = a.GetMaxValue();
        start = a.GetStartTime();
        len = static_cast<float>(a.GetLength());
    } else if (std::holds_alternative<animation::StringAnimation>(anim)) {
        auto& a = std::get<animation::StringAnimation>(anim);
        start = a.GetStartTime();
        len = static_cast<float>(a.GetLength());
        min_val = -region.GetHeight() * 0.025f;
        max_val = region.GetHeight() * 0.025f;
    }
    if (len > 1) {
        custom_zoom.x = 0.9f * region.GetWidth() / len;
        custom_zoom.y = 0.9f * region.GetHeight() / (max_val - min_val);
        const auto h_start = static_cast<float>(-start);
        const auto v_start = max_val;
        canvas.SetView(ImVec2(h_start * custom_zoom.x + 0.05f * region.GetWidth(),
                           v_start * custom_zoom.y + 0.05f * region.GetHeight()),
            1.0f);
    } else {
        // TODO what do you want to see here...
    }
}

void AnimationEditor::SelectAnimation(int32_t a) {
    selectedAnimation = a;
    selectedFloatKey = nullptr;
    selectedStringKey = nullptr;
}

void AnimationEditor::DrawParams() {
    ImGui::Text("Available Parameters");
    ImGui::BeginChild(
        "anim_params", ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2.5f), true);
    for (int32_t a = 0; a < allAnimations.size(); ++a) {
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_SpanFullWidth |
                                   ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_FramePadding |
                                   ImGuiTreeNodeFlags_AllowItemOverlap;
        if (selectedAnimation == a) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }
        const auto& anim = allAnimations[a];
        std::visit([&](auto&& arg) -> void { ImGui::TreeNodeEx(arg.GetName().c_str(), flags); }, anim);
        //ImGui::TreeNodeEx(anim.GetName().c_str(), flags);
        if (ImGui::IsItemActivated()) {
            SelectAnimation(a);
            if (canvas_visible) {
                CenterAnimation(allAnimations[selectedAnimation]);
            }
        }
    }
    ImGui::EndChild();

    ImGui::Text("Animation");
    ImGui::BeginChild(
        "anim_props", ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2.5f), true);
    if (ImGui::Button("start")) {
        current_frame = animation_bounds[0];
    }
    ImGui::SameLine();
    if (ImGui::Button("reverse")) {
        playing = -1;
        accumulated_ms = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("stop")) {
        playing = 0;
    }
    ImGui::SameLine();
    if (ImGui::Button("play")) {
        playing = 1;
        accumulated_ms = 0.0f;
    }
    ImGui::SameLine();
    if (ImGui::Button("end")) {
        current_frame = animation_bounds[1];
    }
    if (ImGui::InputInt("Playback fps target", &playback_fps)) {
        targeted_frame_time = 1000.0f / static_cast<float>(playback_fps);
    }
    if (ImGui::InputInt2("Active Region", animation_bounds, ImGuiInputTextFlags_EnterReturnsTrue)) {
        current_frame = std::clamp(current_frame, animation_bounds[0], animation_bounds[1]);
    }
    ImGui::SliderInt("Current Frame", &current_frame, animation_bounds[0], animation_bounds[1]);
    ImGui::EndChild();
}


void AnimationEditor::DrawInterpolation(
    ImDrawList* dl, const animation::FloatKey& key, const animation::FloatKey& key2) {
    const auto line_col = ImGui::GetColorU32(ImGuiCol_NavHighlight);
    const auto reference_col = IM_COL32(255, 0, 0, 255);
    auto drawList = ImGui::GetWindowDrawList();
    auto pos = ImVec2(key.time, key.value * -1.0f) * custom_zoom;
    auto pos2 = ImVec2(key2.time, key2.value * -1.0f) * custom_zoom;
    switch (key.interpolation) {
    case animation::InterpolationType::Step:
        drawList->AddLine(pos, ImVec2(pos2.x, pos.y), line_col);
        drawList->AddLine(ImVec2(pos2.x, pos.y), pos2, line_col);
        break;
    case animation::InterpolationType::Linear:
        drawList->AddLine(pos, pos2, line_col);
        break;
    case animation::InterpolationType::Hermite:
    case animation::InterpolationType::CubicBezier: {
        // draw reference
        auto step = 0.02f;
        for (auto f = 0.0f; f < 1.0f; f += step) {
            auto v1 = key.Interpolate(key, key2, f);
            v1.y *= -1.0f;
            auto v2 = key.Interpolate(key, key2, f + step);
            v2.y *= -1.0f;
            drawList->AddLine(v1 * custom_zoom, v2 * custom_zoom, reference_col);
        }
        for (auto t = key.time; t < key2.time; ++t) {
            auto v1 = key.Interpolate(key, key2, t);
            auto v2 = key.Interpolate(key, key2, t + 1);
            drawList->AddLine(ImVec2(t, v1 * -1.0f) * custom_zoom, ImVec2(t + 1, v2 * -1.0f) * custom_zoom, line_col);
        }
        break;
    }
    default:;
    }
}


void AnimationEditor::DrawFloatKey(ImDrawList* dl, animation::FloatKey& key) {
    const float size = 4.0f;
    const ImVec2 button_size = {8.0f, 8.0f};
    auto key_color = IM_COL32(255, 128, 0, 255);
    auto active_key_color = IM_COL32(255, 192, 96, 255);
    auto tangent_color = IM_COL32(255, 255, 0, 255);

    auto drawList = ImGui::GetWindowDrawList();

    auto time = key.time;
    auto pos = ImVec2(time, key.value * -1.0f) * custom_zoom;
    auto t_out = ImVec2(time + key.out_tangent.x, (key.value + key.out_tangent.y) * -1.0f) * custom_zoom;

    const auto t_in = ImVec2(time + key.in_tangent.x, (key.value + key.in_tangent.y) * -1.0f) * custom_zoom;
    ImGui::SetCursorScreenPos(ImVec2{t_in.x - (button_size.x / 2.0f), t_in.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key_intan") + std::to_string(key.time)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        curr_interaction = InteractionType::DraggingLeftTangent;
        drag_start = key.in_tangent;
        draggingFloatKey = &key;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingLeftTangent &&
        draggingFloatKey == &key) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        key.in_tangent = drag_start + ImVec2(delta.x / custom_zoom.x, -1.0f * (delta.y / custom_zoom.y));
        if (key.tangents_linked) {
            key.out_tangent = ImVec2(-key.in_tangent.x, -key.in_tangent.y);
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingLeftTangent &&
        &key == draggingFloatKey) {
        curr_interaction = InteractionType::None;
    }
    drawList->AddLine(t_in, pos, tangent_color);
    drawList->AddCircleFilled(t_in, size, tangent_color, 4);

    ImGui::SetCursorScreenPos(ImVec2{t_out.x - (button_size.x / 2.0f), t_out.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key_outtan") + std::to_string(key.time)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        curr_interaction = InteractionType::DraggingRightTangent;
        drag_start = key.out_tangent;
        draggingFloatKey = &key;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingRightTangent &&
        draggingFloatKey == &key) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        key.out_tangent = drag_start + ImVec2(delta.x / custom_zoom.x, -1.0f * (delta.y / custom_zoom.y));
        if (key.tangents_linked) {
            key.in_tangent = ImVec2(-key.out_tangent.x, -key.out_tangent.y);
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingRightTangent &&
        &key == draggingFloatKey) {
        curr_interaction = InteractionType::None;
    }
    drawList->AddLine(pos, t_out, tangent_color);
    drawList->AddCircleFilled(t_out, size, tangent_color, 4);

    if (selectedFloatKey == &key) {
        drawList->AddCircleFilled(pos, size, active_key_color);
    } else {
        drawList->AddCircleFilled(pos, size, key_color);
    }

    ImGui::SetCursorScreenPos(ImVec2{pos.x - (button_size.x / 2.0f), pos.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key") + std::to_string(key.time)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        selectedFloatKey = &key;
        drag_start_value = selectedFloatKey->value;
        drag_start_time = selectedFloatKey->time;
        curr_interaction = InteractionType::DraggingKey;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        selectedFloatKey->value = drag_start_value - delta.y / custom_zoom.y;
        auto new_time = drag_start_time + static_cast<animation::KeyTimeType>(delta.x / custom_zoom.x);
        if (new_time != selectedFloatKey->time) {
            auto& anim = std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]);
            if (anim.HasKey(new_time)) {
                // space is occupied. we do not want that.
            } else {
                selectedFloatKey->time = new_time;
            }
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey &&
        &key == selectedFloatKey) {
        if (drag_start_time != selectedFloatKey->time) {
            // fix sorting
            auto k = *selectedFloatKey;
            auto& anim = std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]);
            anim.DeleteKey(drag_start_time);
            anim.AddKey(k);
            selectedFloatKey = &anim[k.time];
        }
        curr_interaction = InteractionType::None;
    }
    // below here, do not touch key anymore, it might have been nuked by the sorting above
}


void AnimationEditor::DrawPlayhead(ImDrawList* drawList) {
    const ImVec2 playhead_size = {12.0f, 12.0f};
    auto playhead_color = IM_COL32(255, 128, 0, 255);
    auto ph_pos = ImVec2(current_frame * custom_zoom.x, -canvas.ViewOrigin().y);
    ImGui::SetCursorScreenPos(ImVec2(ph_pos.x - playhead_size.x, ph_pos.y));
    ImGui::InvisibleButton("##playhead", playhead_size * 2.0f);
    if (ImGui::IsItemActivated()) {
        curr_interaction = InteractionType::DraggingPlayhead;
        drag_start_time = current_frame;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingPlayhead) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        current_frame = drag_start_time + static_cast<animation::KeyTimeType>(delta.x / custom_zoom.x);
        WriteValuesToGraph();
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingPlayhead) {
        curr_interaction = InteractionType::None;
    }
    drawList->AddTriangleFilled(ImVec2(ph_pos.x - playhead_size.x, ph_pos.y),
        ImVec2(ph_pos.x + playhead_size.x, ph_pos.y), ImVec2(ph_pos.x, ph_pos.y + playhead_size.y), playhead_color);
    drawList->AddLine(ph_pos, ImVec2(ph_pos.x, -canvas.ViewOrigin().y + canvas.Rect().GetHeight()), playhead_color);
}


void AnimationEditor::DrawStringKey(ImDrawList* im_draws, animation::StringKey& key) {
    const float size = 4.0f;
    const ImVec2 button_size = {8.0f, 8.0f};
    auto key_color = IM_COL32(255, 128, 0, 255);
    auto active_key_color = IM_COL32(255, 192, 96, 255);
    auto tangent_color = IM_COL32(255, 255, 0, 255);

    auto drawList = ImGui::GetWindowDrawList();

    auto time = key.time;
    auto pos = ImVec2(time, 0.0f) * custom_zoom;

    drawList->AddText(pos + button_size, key_color, key.value.c_str());
    if (selectedStringKey == &key) {
        drawList->AddCircleFilled(pos, size, active_key_color);
    } else {
        drawList->AddCircleFilled(pos, size, key_color);
    }

    ImGui::SetCursorScreenPos(ImVec2{pos.x - (button_size.x / 2.0f), pos.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key") + std::to_string(key.time)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        selectedStringKey = &key;
        drag_start_time = selectedStringKey->time;
        curr_interaction = InteractionType::DraggingKey;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        auto new_time = drag_start_time + static_cast<animation::KeyTimeType>(delta.x / custom_zoom.x);
        if (new_time != selectedStringKey->time) {
            auto& anim = std::get<animation::StringAnimation>(allAnimations[selectedAnimation]);
            if (anim.HasKey(new_time)) {
                // space is occupied. we do not want that.
            } else {
                selectedStringKey->time = new_time;
            }
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey &&
        &key == selectedStringKey) {
        if (drag_start_time != selectedStringKey->time) {
            // fix sorting
            auto k = *selectedStringKey;
            auto& anim = std::get<animation::StringAnimation>(allAnimations[selectedAnimation]);
            anim.DeleteKey(drag_start_time);
            anim.AddKey(k);
            selectedStringKey = &anim[k.time];
        }
        curr_interaction = InteractionType::None;
    }
    // below here, do not touch key anymore, it might have been nuked by the sorting above
}


void AnimationEditor::DrawCurves() {
    ImGui::Text("");
    ImGui::BeginChild("anim_curves", ImGui::GetContentRegionAvail(), true);
    canvas_visible = canvas.Begin("anim_curves", ImGui::GetContentRegionAvail());
    if (canvas_visible) {
        auto drawList = ImGui::GetWindowDrawList();

        if ((is_dragging || ImGui::IsItemHovered()) && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
            if (!is_dragging) {
                is_dragging = true;
                drag_start = canvas.ViewOrigin();
            }
            canvas.SetView(drag_start + ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0.0f), 1.0f);
        } else if (is_dragging) {
            is_dragging = false;
        }

        DrawGrid();
        DrawScale();

        if (selectedAnimation != -1) {
            if (std::holds_alternative<animation::FloatAnimation>(allAnimations[selectedAnimation])) {
                auto& anim = std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]);
                if (anim.GetSize() > 0) {
                    auto keys = anim.GetAllKeys();
                    for (auto i = 0; i < keys.size(); ++i) {
                        auto& k = anim[keys[i]];
                        if (i < keys.size() - 1) {
                            auto& k2 = anim[keys[i + 1]];
                            DrawInterpolation(drawList, k, k2);
                        }
                        DrawFloatKey(drawList, k);
                    }
                }
            } else if (std::holds_alternative<animation::StringAnimation>(allAnimations[selectedAnimation])) {
                auto& anim = std::get<animation::StringAnimation>(allAnimations[selectedAnimation]);
                if (anim.GetSize() > 0) {
                    auto keys = anim.GetAllKeys();
                    for (auto i = 0; i < keys.size(); ++i) {
                        auto& k = anim[keys[i]];
                        DrawStringKey(drawList, k);
                    }
                }
            }
        }

        DrawPlayhead(drawList);

        canvas.End();


        if (ImGui::IsItemHovered() && isfinite(gui_mouse_wheel) && gui_mouse_wheel != 0) {
            float factor = 1.0f;
            auto old_zoom = custom_zoom;
            if (gui_mouse_wheel > 0) {
                factor = 1.25f * gui_mouse_wheel;
            } else {
                factor = 0.8f * std::abs(gui_mouse_wheel);
            }
            if (ImGui::IsKeyDown(ImGuiKey_ModShift)) {
                custom_zoom.y *= factor;
            } else if (ImGui::IsKeyDown(ImGuiKey_ModCtrl)) {
                custom_zoom.x *= factor;
            } else {
                custom_zoom *= factor;
            }
            auto widget_coords = canvas.ToLocal(ImGui::GetMousePos());
            auto old_point = widget_coords / old_zoom;
            auto mouse_pos = ImGui::GetMousePos() - canvas.Rect().GetTL();
            auto new_origin = (mouse_pos - old_point * custom_zoom) / custom_zoom;
            // these are the invariants
            //auto old_point_screen_reconstructed = (old_point + old_orig) * old_zoom;
            //auto new_point_screen_reconstructed = (old_point + new_origin) * custom_zoom;
            canvas.SetView(new_origin * custom_zoom, 1.0);
        }
    }
    ImGui::EndChild();
}


void AnimationEditor::DrawProperties() {
    ImGui::Text("Properties");
    ImGui::BeginChild(
        "key_props", ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2.5f), true);
    if (selectedAnimation > -1) {
        auto anim = allAnimations[selectedAnimation];
        if (std::holds_alternative<animation::FloatAnimation>(anim) && selectedFloatKey != nullptr) {
            ImGui::InputInt("Time", &selectedFloatKey->time);
            ImGui::InputFloat("Value", &selectedFloatKey->value);
            const char* items[] = {"Step", "Linear", "Hermite", "Cubic Bezier"};
            auto current_item = items[static_cast<int32_t>(selectedFloatKey->interpolation)];
            if (ImGui::BeginCombo("Interpolation", current_item)) {
                for (int n = 0; n < 4; n++) {
                    const bool is_selected = (current_item == items[n]);
                    if (ImGui::Selectable(items[n], is_selected)) {
                        current_item = items[n];
                        switch (n) {
                        case 0:
                            selectedFloatKey->interpolation = animation::InterpolationType::Step;
                            break;
                        case 1:
                            selectedFloatKey->interpolation = animation::InterpolationType::Linear;
                            break;
                        case 2:
                            selectedFloatKey->interpolation = animation::InterpolationType::Hermite;
                            break;
                        case 3:
                            selectedFloatKey->interpolation = animation::InterpolationType::CubicBezier;
                            break;
                        }
                    }
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
        } else if (std::holds_alternative<animation::StringAnimation>(anim) && selectedStringKey != nullptr) {
            ImGui::InputInt("Time", &selectedStringKey->time);
            ImGui::InputText("Value", &selectedStringKey->value);
        }
    }
    ImGui::EndChild();
}


void AnimationEditor::DrawVerticalSeparator() {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::Button(" ", ImVec2(2, 0));
    ImGui::PopItemFlag();
}


void AnimationEditor::DrawGrid() {
    auto drawList = ImGui::GetWindowDrawList();
    const auto top = canvas.ToLocal(ImVec2(0.0f, 0.0f));
    const auto bottom = canvas.ToLocal(canvas.Rect().GetBR());
    auto minorColor = ImGui::GetColorU32(ImGuiCol_Border);
    auto majorColor = ImGui::GetColorU32(ImGuiCol_Text);
    auto textColor = ImGui::GetColorU32(ImGuiCol_Text);
    
    auto region_start = ImVec2(-canvas.ViewOrigin().x, - canvas.Rect().GetHeight() + canvas.ViewOrigin().y);
    auto region_end = ImVec2(canvas.Rect().GetWidth() - canvas.ViewOrigin().x, canvas.ViewOrigin().y);
    region_start /= custom_zoom;
    region_end /= custom_zoom;
    auto width = region_end.x - region_start.x;

    auto dist_order = round(std::log10(width));
    auto majorUnit = std::powf(10, dist_order - 1.0f);
    auto minorUnit = majorUnit / 10.0f;

    auto minor_start = std::nearbyintf(region_start.x / minorUnit) * minorUnit;
    minor_start = std::min(std::max<float>(minor_start, animation_bounds[0]), region_end.x);
    auto minor_end = std::max(std::min<float>(region_end.x, animation_bounds[1]), minor_start);
    for (auto s = minor_start; s < minor_end; s += minorUnit) {
        drawList->AddLine(ImVec2(s * custom_zoom.x, top.y), ImVec2(s * custom_zoom.x, bottom.y), minorColor);
    }

    auto major_start = std::nearbyintf(region_start.x / majorUnit) * majorUnit;
    for (auto s = major_start; s < region_end.x; s += majorUnit) {
        drawList->AddLine(ImVec2(s * custom_zoom.x, top.y), ImVec2(s * custom_zoom.x, bottom.y), majorColor);

        char label[16];
        snprintf(label, 15, "%g", s);
        auto labelSize = ImGui::CalcTextSize(label);

        auto labelPosition = ImVec2((s + minorUnit) * custom_zoom.x, bottom.y - 2.0f * 8.0f);
        drawList->AddText(labelPosition, textColor, label);
    }
}

void AnimationEditor::DrawScale() {

    auto drawList = ImGui::GetWindowDrawList();
    const auto top = canvas.ToLocal(ImVec2(0.0f, 0.0f));
    const auto bottom = canvas.ToLocal(canvas.Rect().GetBR());
    auto minorColor = ImGui::GetColorU32(ImGuiCol_Border);
    auto majorColor = ImGui::GetColorU32(ImGuiCol_Text);
    auto textColor = ImGui::GetColorU32(ImGuiCol_Text);
    const auto minorSize = 10.0f;
    const auto majorSize = 15.0f;

    auto region_start = ImVec2(-canvas.ViewOrigin().x, - canvas.Rect().GetHeight() + canvas.ViewOrigin().y);
    auto region_end = ImVec2(canvas.Rect().GetWidth() - canvas.ViewOrigin().x, canvas.ViewOrigin().y);
    region_start /= custom_zoom;
    region_end /= custom_zoom;
    auto height = region_end.y - region_start.y;

    auto dist_order = round(std::log10(height));
    auto majorUnit = std::powf(10, dist_order - 1.0f);
    auto minorUnit = majorUnit / 10.0f;

    auto minor_start = std::nearbyintf(region_start.y / minorUnit) * minorUnit;
    auto ref_pt =
        ImVec2(std::max(region_start.x + minorSize, 0.0f * custom_zoom.x), 0.0f);
    ref_pt.x = std::min(ref_pt.x,  region_end.x - minorSize);
    for (auto s = minor_start; s < region_end.y; s += minorUnit) {
        drawList->AddLine(ref_pt + ImVec2(-minorSize, -s * custom_zoom.y), ref_pt + ImVec2(minorSize, -s * custom_zoom.y), minorColor);
    }

    auto major_start = std::nearbyintf(region_start.y / majorUnit) * majorUnit;
    for (auto s = major_start; s < region_end.y; s += majorUnit) {
        drawList->AddLine(ref_pt + ImVec2(-minorSize, -s * custom_zoom.y), ref_pt + ImVec2(minorSize, -s * custom_zoom.y), majorColor);

        char label[16];
        snprintf(label, 15, "%g", s);
        auto labelSize = ImGui::CalcTextSize(label);

        auto labelPosition = ImVec2(ref_pt + ImVec2(minorSize, -s * custom_zoom.y));
        drawList->AddText(labelPosition, textColor, label);
    }
}

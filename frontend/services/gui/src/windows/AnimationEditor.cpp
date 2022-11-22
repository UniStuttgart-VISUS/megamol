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
        if (param_slot->Param<FloatParam>() || param_slot->Param<IntParam>()) {
            const float the_val = std::stof(new_value);
            bool found = false;
            for (auto& a : floatAnimations) {
                if (a.GetName() == the_name) {
                    if (a.HasKey(current_frame)) {
                        a[current_frame].value = the_val;
                    } else {
                        a.AddKey({current_frame, the_val});
                    }
                    found = true;
                }
            }
            if (!found) {
                animation::FloatAnimation f = {the_name};
                f.AddKey({current_frame, the_val});
                floatAnimations.push_back(f);
            }
        }
        if (param_slot->Param<EnumParam>() || param_slot->Param<FlexEnumParam>() || param_slot->Param<StringParam>() ||
            param_slot->Param<FilePathParam>()) {
            bool found = false;
            //for (auto& a : stringAnimations) {
            //
            //}
        }
    }
    // TODO: what else. Enum probably.
    return true;
}


void megamol::gui::AnimationEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {}


void megamol::gui::AnimationEditor::SpecificStateToJSON(nlohmann::json& inout_json) {}


void AnimationEditor::WriteValuesToGraph() {
    if (write_to_graph) {
        std::stringstream lua_commands;
        for (auto a : floatAnimations) {
            lua_commands << "mmSetParamValue(\"" << a.GetName() << "\",[=[" << a.GetValue(current_frame) << "]=])\n";
        }
        auto res = (*input_lua_func)(lua_commands.str());
        if (!std::get<0>(res)) {
            open_popup_error = true;
            error_popup_message = std::get<1>(res);
            playback_active = false;
        }
    }
}


bool AnimationEditor::SaveToFile(const std::string& file) {
    nlohmann::json animation_data;
    nlohmann::json anims;
    for (auto& fa : floatAnimations) {
        anims.push_back(fa);
    }
    animation_data["float_animations"] = anims;
    std::ofstream out(file);
    if (!out.is_open()) {
        return false;
    }
    auto s = animation_data.dump(2);
    out << s;
    out.close();
    return true;
}


void AnimationEditor::ClearData() {
    floatAnimations.clear();
    selectedAnimation = -1;
    selectedKey = nullptr;
    animation_file = "";
}

bool AnimationEditor::LoadFromFile(std::string file) {
    ClearData();
    std::ifstream in(file);
    if (!in.is_open()) {
        return false;
    }
    auto j_all = nlohmann::json::parse(in);
    animation::FloatAnimation dummy("dummy");
    for (auto& j : j_all["float_animations"]) {
        // j.get<FloatAnimation>() does not work because no default constructor
        from_json(j, dummy);
        floatAnimations.push_back(dummy);
    }
    animation_file = file;
    return true;
}


void AnimationEditor::DrawPopups() {
    if (this->file_browser.PopUp_Load(
            "Load Animation", animation_file, open_popup_load, {"anim"}, FilePathParam::Flag_File_RestrictExtension)) {
        if (!LoadFromFile(animation_file)) {
            error_popup_message = "Saving failed.";
            open_popup_error = true;
        }
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
    ImGui::SameLine();
    if (ImGui::Button("break tangents")) {
        if (selectedKey != nullptr) {
            selectedKey->tangents_linked = false;
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("link tangents")) {
        if (selectedKey != nullptr) {
            selectedKey->tangents_linked = true;
            selectedKey->out_tangent = ImVec2(-selectedKey->in_tangent.x, -selectedKey->in_tangent.y);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("flat tangents")) {
        if (selectedKey != nullptr) {
            selectedKey->in_tangent = {-1.0f, 0.0f};
            selectedKey->out_tangent = {1.0f, 0.0f};
        }
    }
    ImGui::SameLine();
    DrawVerticalSeparator();
    ImGui::SameLine();
    if (ImGui::Button("frame view")) {
        center_animation(floatAnimations[selectedAnimation]);
    }
    ImGui::SameLine();
    DrawVerticalSeparator();
    ImGui::SameLine();
    ImGui::PushItemWidth(100.0f);
    ImGui::SliderFloat("HZoom", &custom_zoom.x, 0.01f, 5000.0f);
    ImGui::SameLine();
    ImGui::SliderFloat("VZoom", &custom_zoom.y, 0.01f, 5000.0f);
}


void AnimationEditor::center_animation(const animation::FloatAnimation& anim) {
    auto region = canvas.Rect();
    if (anim.GetLength() > 1) {
        custom_zoom.x = 0.9f * region.GetWidth() / static_cast<float>(anim.GetLength());
        custom_zoom.y = 0.9f * region.GetHeight() / (anim.GetMaxValue() - anim.GetMinValue());
        const auto h_start = static_cast<float>(-anim.GetStartTime());
        const auto v_start = anim.GetMaxValue();
        canvas.SetView(ImVec2(h_start * custom_zoom.x + 0.05f * region.GetWidth(),
                           v_start * custom_zoom.y + 0.05f * region.GetHeight()),
            1.0f);
    } else {
        // TODO what do you want to see here...
    }
}

void AnimationEditor::DrawParams() {
    ImGui::Text("Available Parameters");
    ImGui::BeginChild(
        "anim_params", ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2.5f), true);
    for (int32_t a = 0; a < floatAnimations.size(); ++a) {
        const auto& anim = floatAnimations[a];
        ImGuiTreeNodeFlags flags = ImGuiTreeNodeFlags_Leaf | ImGuiTreeNodeFlags_SpanFullWidth |
                                   ImGuiTreeNodeFlags_NoTreePushOnOpen | ImGuiTreeNodeFlags_FramePadding |
                                   ImGuiTreeNodeFlags_AllowItemOverlap;
        if (selectedAnimation == a) {
            flags |= ImGuiTreeNodeFlags_Selected;
        }
        ImGui::TreeNodeEx(anim.GetName().c_str(), flags);
        if (ImGui::IsItemActivated()) {
            selectedAnimation = a;
            if (canvas_visible) {
                center_animation(floatAnimations[selectedAnimation]);
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
    ImGui::Button("reverse");
    ImGui::SameLine();
    ImGui::Button("play");
    ImGui::SameLine();
    if (ImGui::Button("end")) {
        current_frame = animation_bounds[1];
    }
    if (ImGui::InputInt2("Active Region", animation_bounds)) {
        current_frame = std::clamp(current_frame, animation_bounds[0], animation_bounds[1]);
    }
    ImGui::SliderInt("Current Frame", &current_frame, animation_bounds[0], animation_bounds[1]);
    ImGui::EndChild();
}


void AnimationEditor::DrawInterpolation(ImDrawList* dl, const animation::FloatKey& key, const animation::FloatKey& key2) {
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


void AnimationEditor::DrawKey(ImDrawList* dl, animation::FloatKey& key) {
    const float size = 4.0f;
    const ImVec2 button_size = {8.0f, 8.0f};
    auto key_color = IM_COL32(255, 128, 0, 255);
    auto active_key_color = IM_COL32(255, 192, 96, 255);
    auto tangent_color = IM_COL32(255, 255, 0, 255);

    auto drawList = ImGui::GetWindowDrawList();

    auto time = key.time;
    auto pos = ImVec2(time, key.value * -1.0f) * custom_zoom;
    auto t_out = ImVec2(time + key.out_tangent.x, (key.value + key.out_tangent.y) * -1.0f) * custom_zoom;
    drawList->AddLine(pos, t_out, tangent_color);

    const auto t_in = ImVec2(time + key.in_tangent.x, (key.value + key.in_tangent.y) * -1.0f) * custom_zoom;
    ImGui::SetCursorScreenPos(ImVec2{t_in.x - (button_size.x / 2.0f), t_in.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key_intan") + std::to_string(key.time)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        curr_interaction = InteractionType::DraggingLeftTangent;
        drag_start = key.in_tangent;
        draggingKey = &key;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingLeftTangent &&
        draggingKey == &key) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        key.in_tangent = drag_start + ImVec2(delta.x / custom_zoom.x, -1.0f * (delta.y / custom_zoom.y));
        if (key.tangents_linked) {
            key.out_tangent = ImVec2(-key.in_tangent.x, -key.in_tangent.y);
        }
    }
    drawList->AddLine(t_in, pos, tangent_color);
    drawList->AddCircleFilled(t_in, size, tangent_color, 4);
    if (ImGui::IsItemDeactivated()) {
        curr_interaction = InteractionType::None;
    }

    ImGui::SetCursorScreenPos(ImVec2{t_out.x - (button_size.x / 2.0f), t_out.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key_outtan") + std::to_string(key.time)).c_str(), button_size);
    drawList->AddCircleFilled(t_out, size, tangent_color, 4);
    if (ImGui::IsItemActivated()) {
        curr_interaction = InteractionType::DraggingRightTangent;
        drag_start = key.out_tangent;
        draggingKey = &key;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingRightTangent &&
        draggingKey == &key) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        key.out_tangent = drag_start + ImVec2(delta.x / custom_zoom.x, -1.0f * (delta.y / custom_zoom.y));
        if (key.tangents_linked) {
            key.in_tangent = ImVec2(-key.out_tangent.x, -key.out_tangent.y);
        }
    }
    if (ImGui::IsItemDeactivated()) {
        curr_interaction = InteractionType::None;
    }

    ImGui::SetCursorScreenPos(ImVec2{pos.x - (button_size.x / 2.0f), pos.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key") + std::to_string(key.time)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        selectedKey = &key;
        drag_start_value = selectedKey->value;
        drag_start_time = selectedKey->time;
        curr_interaction = InteractionType::DraggingKey;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        selectedKey->value = drag_start_value - delta.y / custom_zoom.y;
        selectedKey->time = drag_start_time + static_cast<animation::KeyTimeType>(delta.x / custom_zoom.x);
    }
    if (selectedKey == &key) {
        drawList->AddCircleFilled(pos, size, active_key_color);
    } else {
        drawList->AddCircleFilled(pos, size, key_color);
    }
    if (ImGui::IsItemDeactivated()) {
        curr_interaction = InteractionType::None;
    }
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
    drawList->AddTriangleFilled(ImVec2(ph_pos.x - playhead_size.x, ph_pos.y),
        ImVec2(ph_pos.x + playhead_size.x, ph_pos.y), ImVec2(ph_pos.x, ph_pos.y + playhead_size.y), playhead_color);
    drawList->AddLine(ph_pos, ImVec2(ph_pos.x, canvas.Rect().GetHeight()), playhead_color);
    if (ImGui::IsItemDeactivated()) {
        curr_interaction = InteractionType::None;
    }
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

        const auto viewRect = canvas.ViewRect();
        if (viewRect.Max.x > 0.0f) {
            DrawGrid(ImVec2(0.0f, 0.0f), ImVec2(viewRect.Max.x, 0.0f), 100.0f, 10.0f, 0.6f);
        }
        if (viewRect.Max.y > 0.0f) {
            DrawScale(ImVec2(0.0f, 0.0f), ImVec2(0.0f, viewRect.Max.y), 100.0f, 10.0f, 0.6f, -1.0f);
        }
        if (viewRect.Min.y < 0.0f) {
            DrawScale(ImVec2(0.0f, 0.0f), ImVec2(0.0f, viewRect.Min.y), 100.0f, 10.0f, 0.6f);
        }

        if (selectedAnimation != -1) {
            auto& anim = floatAnimations[selectedAnimation];
            if (anim.GetSize() > 0) {
                auto keys = anim.GetAllKeys();
                for (auto i = 0; i < keys.size(); ++i) {
                    auto& k = anim[keys[i]];
                    if (i < keys.size() - 1) {
                        auto& k2 = anim[keys[i + 1]];
                        DrawInterpolation(drawList, k, k2);
                    }
                    DrawKey(drawList, k);
                }
            }
        }

        DrawPlayhead(drawList);

        canvas.End();
    }
    ImGui::EndChild();
}


void AnimationEditor::DrawProperties() {
    ImGui::Text("Properties");
    ImGui::BeginChild(
        "key_props", ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2.5f), true);
    if (selectedAnimation > -1 && selectedKey != nullptr) {
        ImGui::InputInt("Time", &selectedKey->time);
        ImGui::InputFloat("Value", &selectedKey->value);
        const char* items[] = {"Step", "Linear", "Hermite", "Cubic Bezier"};
        auto current_item = items[static_cast<int32_t>(selectedKey->interpolation)];
        if (ImGui::BeginCombo("Interpolation", current_item)) {
            for (int n = 0; n < 4; n++) {
                const bool is_selected = (current_item == items[n]);
                if (ImGui::Selectable(items[n], is_selected)) {
                    current_item = items[n];
                    switch (n) {
                    case 0:
                        selectedKey->interpolation = animation::InterpolationType::Step;
                        break;
                    case 1:
                        selectedKey->interpolation = animation::InterpolationType::Linear;
                        break;
                    case 2:
                        selectedKey->interpolation = animation::InterpolationType::Hermite;
                        break;
                    case 3:
                        selectedKey->interpolation = animation::InterpolationType::CubicBezier;
                        break;
                    }
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }
    }
    ImGui::EndChild();
}


void AnimationEditor::DrawVerticalSeparator() {
    ImGui::PushItemFlag(ImGuiItemFlags_Disabled, true);
    ImGui::Button(" ", ImVec2(2, 0));
    ImGui::PopItemFlag();
}


void AnimationEditor::DrawGrid(
    const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment, float sign) {

    // TODO only show active region
    // TODO auto compute Units based on zoom
    auto drawList = ImGui::GetWindowDrawList();
    auto direction = (to - from) * ImInvLength(to - from, 0.0f);
    auto normal = ImVec2(-direction.y, direction.x);
    auto distance = sqrtf(ImLengthSqr(to - from));

    if (ImDot(direction, direction) < FLT_EPSILON)
        return;

    auto labelDistance = 8.0f;

    auto minorColor = ImGui::GetColorU32(ImGuiCol_Border);
    auto textColor = ImGui::GetColorU32(ImGuiCol_Text);

    drawList->AddLine(from, to, IM_COL32(255, 255, 255, 255));

    auto p = from * custom_zoom;
    const auto top = canvas.ToLocal(ImVec2(0.0f, 0.0f));
    const auto bottom = canvas.ToLocal(canvas.Rect().GetBR());
    for (auto d = 0.0f; d <= distance;
         d += minorUnit * ImDot(direction, custom_zoom), p += direction * minorUnit * custom_zoom) {
        drawList->AddLine(ImVec2(p.x, top.y), ImVec2(p.x, bottom.y), minorColor);
    }

    for (auto d = 0.0f; d <= distance + majorUnit; d += majorUnit) {
        p = from + direction * d;
        p *= custom_zoom;

        drawList->AddLine(ImVec2(p.x, top.y), ImVec2(p.x, bottom.y), IM_COL32(255, 255, 255, 255));

        if (d == 0.0f)
            continue;

        char label[16];
        snprintf(label, 15, "%g", d * sign);
        auto labelSize = ImGui::CalcTextSize(label);

        auto labelPosition = p + ImVec2(fabsf(normal.x), fabsf(normal.y)) * labelDistance;
        labelPosition.y = bottom.y - 2.0f * labelDistance;
        auto labelAlignedSize = ImDot(labelSize, direction);
        labelPosition += direction * (-labelAlignedSize + labelAlignment * labelAlignedSize * 2.0f);
        labelPosition = ImFloor(labelPosition + ImVec2(0.5f, 0.5f));

        drawList->AddText(labelPosition, textColor, label);
    }
}

void AnimationEditor::DrawScale(
    const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment, float sign) {

    // TODO auto compute Units based on zoom
    auto drawList = ImGui::GetWindowDrawList();
    auto direction = (to - from) * ImInvLength(to - from, 0.0f);
    auto normal = ImVec2(-direction.y, direction.x);
    auto distance = sqrtf(ImLengthSqr(to - from));

    if (ImDot(direction, direction) < FLT_EPSILON)
        return;

    auto minorSize = 5.0f;
    auto majorSize = 10.0f;
    auto labelDistance = 8.0f;

    drawList->AddLine(from, to, IM_COL32(255, 255, 255, 255));

    auto p = from * custom_zoom;
    for (auto d = 0.0f; d <= distance;
         d += minorUnit * ImDot(direction * -sign, custom_zoom), p += direction * minorUnit * custom_zoom)
        drawList->AddLine(p - normal * minorSize, p + normal * minorSize, IM_COL32(255, 255, 255, 255));

    for (auto d = 0.0f; d <= distance + majorUnit; d += majorUnit) {
        p = from + direction * d;
        p *= custom_zoom;

        drawList->AddLine(p - normal * majorSize, p + normal * majorSize, IM_COL32(255, 255, 255, 255));

        if (d == 0.0f)
            continue;

        char label[16];
        snprintf(label, 15, "%g", d * sign);
        auto labelSize = ImGui::CalcTextSize(label);

        auto labelPosition = p + ImVec2(fabsf(normal.x), fabsf(normal.y)) * labelDistance;
        auto labelAlignedSize = ImDot(labelSize, direction);
        labelPosition += direction * (-labelAlignedSize + labelAlignment * labelAlignedSize * 2.0f);
        labelPosition = ImFloor(labelPosition + ImVec2(0.5f, 0.5f));

        drawList->AddText(labelPosition, IM_COL32(255, 255, 255, 255), label);
    }
}

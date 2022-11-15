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


std::map<unsigned, Key>::iterator FloatAnimation::begin() {
    return keys.begin();
}


std::map<unsigned, Key>::iterator FloatAnimation::end() {
    return keys.end();
}


FloatAnimation::ValueType FloatAnimation::GetValue(KeyTimeType time) const {
    if (keys.size() < 2)
        return 0.0f;
    Key before_key = keys.begin()->second, after_key = keys.begin()->second;
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


const std::string& FloatAnimation::GetName() const {
    return param_name;
}


KeyTimeType FloatAnimation::GetStartTime() const {
    if (keys.size() > 0) {
        return keys.begin()->second.time;
    } else {
        return 0;
    }
}


KeyTimeType FloatAnimation::GetEndTime() const {
    if (keys.size() > 0) {
        return keys.rbegin()->second.time;
    } else {
        return 1;
    }
}


FloatAnimation::ValueType FloatAnimation::GetMinValue() const {
    auto min = std::numeric_limits<float>::max();
    for (auto& k: keys) {
        min = std::min(min, k.second.value);
    }
    return min;
}


FloatAnimation::ValueType FloatAnimation::GetMaxValue() const {
    auto max = std::numeric_limits<float>::lowest();
    for (auto& k: keys) {
        max = std::max(max, k.second.value);
    }
    return max;
}


megamol::gui::AnimationEditor::AnimationEditor(const std::string& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_ANIMATIONEDITOR) {

    // Configure HOTKEY EDITOR Window
    this->win_config.size = ImVec2(500.0f * megamol::gui::gui_scaling.Get(), 400.0f * megamol::gui::gui_scaling.Get());
    this->win_config.reset_size = this->win_config.size;
    this->win_config.flags = ImGuiWindowFlags_NoNavInputs;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F5, core::view::Modifier::NONE);

    // TODO we actually want to subscribe to graph updates to be able to "record" parameter changes automatically
    // TODO what do we do with parameters other than floats?
    FloatAnimation f("::renderer::scaling");
    Key k{0, 1.0f, InterpolationType::Linear};
    Key k2{10, 1.5f, InterpolationType::Linear};
    f.AddKey(k);
    f.AddKey(k2);
    floatAnimations.push_back(f);

    FloatAnimation f2("::view::anim::time");
    k.value = 0.0f;
    k2.value = 100.0f;
    f.AddKey(k);
    f.AddKey(k2);
    floatAnimations.push_back(f2);
}


megamol::gui::AnimationEditor::~AnimationEditor() {}


bool megamol::gui::AnimationEditor::Update() {

    return true;
}


bool megamol::gui::AnimationEditor::Draw() {
    DrawToolbar();

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


void megamol::gui::AnimationEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {}


void megamol::gui::AnimationEditor::SpecificStateToJSON(nlohmann::json& inout_json) {}


void AnimationEditor::DrawToolbar() {
    ImGui::Button("some button");
    ImGui::SameLine();
    DrawVerticalSeparator();
    ImGui::SameLine();
    ImGui::Button("break tangents");
    ImGui::SameLine();
    ImGui::Button("link tangents");
    ImGui::SameLine();
    ImGui::Button("flat tangents");
    ImGui::SameLine();
    DrawVerticalSeparator();
    ImGui::SameLine();
    ImGui::Button("some other button");
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
                auto h_center = (anim.GetEndTime() - anim.GetStartTime()) * 0.5f + anim.GetStartTime();
                auto v_center = (anim.GetMaxValue() - anim.GetMinValue()) * 0.5f + anim.GetMinValue();
                canvas.SetView(ImVec2(h_center * frame_width, v_center * value_scale), 1.0f);
            }
        }
    }
    ImGui::EndChild();
}


void AnimationEditor::DrawCurves() {
    ImGui::Text("");
    ImGui::BeginChild("anim_curves", ImGui::GetContentRegionAvail(), true);
    canvas_visible = canvas.Begin("anim_curves", ImGui::GetContentRegionAvail());
    if (canvas_visible) {
        auto canvas_region = ImGui::GetContentRegionAvail();
        auto line_color = ImGui::GetColorU32(ImGuiCol_Border);
        auto text_color = ImGui::GetColorU32(ImGuiCol_Text);
        auto key_color = ImGui::GetColorU32(ImGuiCol_Button);
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        if ((is_dragging || ImGui::IsItemHovered()) && ImGui::IsMouseDragging(ImGuiMouseButton_Right, 0.0f)) {
            if (!is_dragging) {
                is_dragging = true;
                drag_start = canvas.ViewOrigin();
            }
            canvas.SetView(drag_start + ImGui::GetMouseDragDelta(ImGuiMouseButton_Right, 0.0f) * zoom, zoom);
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
            for (auto& k: anim) {
                drawList->AddCircleFilled(ImVec2(k.second.time, k.second.value * value_scale * -1.0f), 4.0f, key_color);
            }
        }
        canvas.End();
    }
    ImGui::EndChild();
}


void AnimationEditor::DrawProperties() {
    ImGui::Text("Properties");
    ImGui::BeginChild(
        "anim_props", ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2.5f), true);
    if (selectedAnimation > -1 && selectedKey != nullptr) {
        // draw property sheet
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
    //auto drawList  = ImGui::GetWindowDrawList();
    //auto direction = (to - from) * ImInvLength(to - from, 0.0f);
    //auto normal    = ImVec2(-direction.y, direction.x);
    //auto distance  = sqrtf(ImLengthSqr(to - from));

    //if (ImDot(direction, direction) < FLT_EPSILON)
    //    return;

    //auto minorSize = 5.0f;
    //auto majorSize = 10.0f;
    //auto labelDistance = 8.0f;

    //drawList->AddLine(from, to, IM_COL32(255, 255, 255, 255));

    //auto p = from;
    //for (auto d = 0.0f; d <= distance; d += minorUnit, p += direction * minorUnit)
    //    drawList->AddLine(p - normal * minorSize, p + normal * minorSize, IM_COL32(255, 255, 255, 255));

    //for (auto d = 0.0f; d <= distance + majorUnit; d += majorUnit)
    //{
    //    p = from + direction * d;

    //    drawList->AddLine(p - normal * majorSize, p + normal * majorSize, IM_COL32(255, 255, 255, 255));

    //    if (d == 0.0f)
    //        continue;

    //    char label[16];
    //    snprintf(label, 15, "%g", d * sign);
    //    auto labelSize = ImGui::CalcTextSize(label);

    //    auto labelPosition    = p + ImVec2(fabsf(normal.x), fabsf(normal.y)) * labelDistance;
    //    auto labelAlignedSize = ImDot(labelSize, direction);
    //    labelPosition += direction * (-labelAlignedSize + labelAlignment * labelAlignedSize * 2.0f);
    //    labelPosition = ImFloor(labelPosition + ImVec2(0.5f, 0.5f));

    //    drawList->AddText(labelPosition, IM_COL32(255, 255, 255, 255), label);
    //}
}

void AnimationEditor::DrawScale(
    const ImVec2& from, const ImVec2& to, float majorUnit, float minorUnit, float labelAlignment, float sign) {
    auto drawList  = ImGui::GetWindowDrawList();
    auto direction = (to - from) * ImInvLength(to - from, 0.0f);
    auto normal    = ImVec2(-direction.y, direction.x);
    auto distance  = sqrtf(ImLengthSqr(to - from));

    if (ImDot(direction, direction) < FLT_EPSILON)
        return;

    auto minorSize = 5.0f;
    auto majorSize = 10.0f;
    auto labelDistance = 8.0f;

    drawList->AddLine(from, to, IM_COL32(255, 255, 255, 255));

    auto p = from;
    for (auto d = 0.0f; d <= distance; d += minorUnit, p += direction * minorUnit)
        drawList->AddLine(p - normal * minorSize, p + normal * minorSize, IM_COL32(255, 255, 255, 255));

    for (auto d = 0.0f; d <= distance + majorUnit; d += majorUnit)
    {
        p = from + direction * d;

        drawList->AddLine(p - normal * majorSize, p + normal * majorSize, IM_COL32(255, 255, 255, 255));

        if (d == 0.0f)
            continue;

        char label[16];
        snprintf(label, 15, "%g", d * sign);
        auto labelSize = ImGui::CalcTextSize(label);

        auto labelPosition    = p + ImVec2(fabsf(normal.x), fabsf(normal.y)) * labelDistance;
        auto labelAlignedSize = ImDot(labelSize, direction);
        labelPosition += direction * (-labelAlignedSize + labelAlignment * labelAlignedSize * 2.0f);
        labelPosition = ImFloor(labelPosition + ImVec2(0.5f, 0.5f));

        drawList->AddText(labelPosition, IM_COL32(255, 255, 255, 255), label);
    }
}

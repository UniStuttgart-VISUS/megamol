/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/animation/AnimationUtils.h"

#include <sstream>

using namespace megamol::core::utility;

void animation::to_json(nlohmann::json& j, const animation::FloatKey& k) {
    j = nlohmann::json{{"time", k.time}, {"value", k.value}, {"tangents_linked", k.tangents_linked},
        {"interpolation", k.interpolation}, {"in_tangent", nlohmann::json{k.in_tangent.x, k.in_tangent.y}},
        {"out_tangent", nlohmann::json{k.out_tangent.x, k.out_tangent.y}}};
}


void animation::from_json(const nlohmann::json& j, animation::FloatKey& k) {
    j["time"].get_to(k.time);
    j["value"].get_to(k.value);
    j["tangents_linked"].get_to(k.tangents_linked);
    j["interpolation"].get_to(k.interpolation);
    j["in_tangent"][0].get_to(k.in_tangent.x);
    j["in_tangent"][1].get_to(k.in_tangent.y);
    j["out_tangent"][0].get_to(k.out_tangent.x);
    j["out_tangent"][1].get_to(k.out_tangent.y);
}


void animation::to_json(nlohmann::json& j, const StringKey& k) {
    j = nlohmann::json{{"time", k.time}, {"value", k.value}};
}


void animation::from_json(const nlohmann::json& j, StringKey& k) {
    j["time"].get_to(k.time);
    j["value"].get_to(k.value);
}


void animation::to_json(nlohmann::json& j, const VectorKey<FloatKey>& k) {
    auto k_array = nlohmann::json::array();
    for (int i = 0; i < k.nestedData.size(); ++i) {
        k_array.push_back(k.nestedData[i]);
    }
    j["nested_data"] = k_array;
}


void animation::from_json(const nlohmann::json& j, VectorKey<FloatKey>& k) {
    auto num = j["nested_data"].size();
    k.nestedData.resize(num);
    for (int i = 0; i < num; ++i) {
        j["nested_data"][i].get_to(k.nestedData[i]);
    }
}


void animation::to_json(nlohmann::json& j, const FloatAnimation& f) {
    j = nlohmann::json{{"name", f.GetName()}, {"type", "float"}};
    auto k_array = nlohmann::json::array();
    for (auto& k : f.GetAllKeys()) {
        k_array.push_back(f[k]);
    }
    j["keys"] = k_array;
}


void animation::from_json(const nlohmann::json& j, FloatAnimation& f) {
    f = FloatAnimation{j.at("name")};
    assert(j.at("type") == "float");
    for (auto& j : j["keys"]) {
        FloatKey k;
        j.get_to(k);
        f.AddKey(k);
    }
}

void animation::to_json(nlohmann::json& j, const StringAnimation& s) {
    j = nlohmann::json{{"name", s.GetName()}, {"type", "string"}};
    auto k_array = nlohmann::json::array();
    for (auto& k : s.GetAllKeys()) {
        k_array.push_back(s[k]);
    }
    j["keys"] = k_array;
}

void animation::from_json(const nlohmann::json& j, StringAnimation& s) {
    s = StringAnimation{j.at("name")};
    assert(j.at("type") == "string");
    for (auto& j : j["keys"]) {
        StringKey k;
        j.get_to(k);
        s.AddKey(k);
    }
}

void animation::to_json(nlohmann::json& j, const FloatVectorAnimation& v) {
    j = nlohmann::json{{"name", v.GetName()}, {"type", "float_vector"}};
    auto v_array = nlohmann::json::array();
    for (auto& k : v.GetAllKeys()) {
        v_array.push_back(v[k]);
    }
    j["keys"] = v_array;
}

void animation::from_json(const nlohmann::json& j, FloatVectorAnimation& v) {
    v = FloatVectorAnimation{j.at("name")};
    assert(j.at("type") == "float_vector");
    for (auto& j : j["keys"]) {
        FloatVectorAnimation::KeyType k;
        j.get_to(k);
        v.AddKey(k);
    }
}

// this does not seem okay
#if 0
static ImVector<ImRect> s_GroupPanelLabelStack;

void animation::BeginGroupPanel(const char* name, const ImVec2& size) {
    ImGui::BeginGroup();

    auto cursorPos = ImGui::GetCursorScreenPos();
    auto itemSpacing = ImGui::GetStyle().ItemSpacing;
    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

    auto frameHeight = ImGui::GetFrameHeight();
    ImGui::BeginGroup();

    ImVec2 effectiveSize = size;
    if (size.x < 0.0f)
        effectiveSize.x = ImGui::GetContentRegionAvail().x;
    else
        effectiveSize.x = size.x;
    ImGui::Dummy(ImVec2(effectiveSize.x, 0.0f));

    ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::BeginGroup();
    ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::TextUnformatted(name);
    auto labelMin = ImGui::GetItemRectMin();
    auto labelMax = ImGui::GetItemRectMax();
    ImGui::SameLine(0.0f, 0.0f);
    ImGui::Dummy(ImVec2(0.0, frameHeight + itemSpacing.y));
    ImGui::BeginGroup();

    //ImGui::GetWindowDrawList()->AddRect(labelMin, labelMax, IM_COL32(255, 0, 255, 255));

    ImGui::PopStyleVar(2);

#if IMGUI_VERSION_NUM >= 17301
    ImGui::GetCurrentWindow()->ContentRegionRect.Max.x -= frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->WorkRect.Max.x -= frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->InnerRect.Max.x -= frameHeight * 0.5f;
#else
    ImGui::GetCurrentWindow()->ContentsRegionRect.Max.x -= frameHeight * 0.5f;
#endif
    ImGui::GetCurrentWindow()->Size.x -= frameHeight;

    auto itemWidth = ImGui::CalcItemWidth();
    ImGui::PushItemWidth(ImMax(0.0f, itemWidth - frameHeight));

    s_GroupPanelLabelStack.push_back(ImRect(labelMin, labelMax));
}

void animation::EndGroupPanel() {
    ImGui::PopItemWidth();

    auto itemSpacing = ImGui::GetStyle().ItemSpacing;

    ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(0.0f, 0.0f));
    ImGui::PushStyleVar(ImGuiStyleVar_ItemSpacing, ImVec2(0.0f, 0.0f));

    auto frameHeight = ImGui::GetFrameHeight();

    ImGui::EndGroup();

    //ImGui::GetWindowDrawList()->AddRectFilled(ImGui::GetItemRectMin(), ImGui::GetItemRectMax(), IM_COL32(0, 255, 0, 64), 4.0f);

    ImGui::EndGroup();

    ImGui::SameLine(0.0f, 0.0f);
    ImGui::Dummy(ImVec2(frameHeight * 0.5f, 0.0f));
    ImGui::Dummy(ImVec2(0.0, frameHeight - frameHeight * 0.5f - itemSpacing.y));

    ImGui::EndGroup();

    auto itemMin = ImGui::GetItemRectMin();
    auto itemMax = ImGui::GetItemRectMax();
    //ImGui::GetWindowDrawList()->AddRectFilled(itemMin, itemMax, IM_COL32(255, 0, 0, 64), 4.0f);

    auto labelRect = s_GroupPanelLabelStack.back();
    s_GroupPanelLabelStack.pop_back();

    ImVec2 halfFrame = ImVec2(frameHeight * 0.25f, frameHeight) * 0.5f;
    ImRect frameRect = ImRect(itemMin + halfFrame, itemMax - ImVec2(halfFrame.x, 0.0f));
    labelRect.Min.x -= itemSpacing.x;
    labelRect.Max.x += itemSpacing.x;
    for (int i = 0; i < 4; ++i) {
        switch (i) {
        // left half-plane
        case 0:
            ImGui::PushClipRect(ImVec2(-FLT_MAX, -FLT_MAX), ImVec2(labelRect.Min.x, FLT_MAX), true);
            break;
        // right half-plane
        case 1:
            ImGui::PushClipRect(ImVec2(labelRect.Max.x, -FLT_MAX), ImVec2(FLT_MAX, FLT_MAX), true);
            break;
        // top
        case 2:
            ImGui::PushClipRect(ImVec2(labelRect.Min.x, -FLT_MAX), ImVec2(labelRect.Max.x, labelRect.Min.y), true);
            break;
        // bottom
        case 3:
            ImGui::PushClipRect(ImVec2(labelRect.Min.x, labelRect.Max.y), ImVec2(labelRect.Max.x, FLT_MAX), true);
            break;
        }

        ImGui::GetWindowDrawList()->AddRect(
            frameRect.Min, frameRect.Max, ImColor(ImGui::GetStyleColorVec4(ImGuiCol_Border)), halfFrame.x);

        ImGui::PopClipRect();
    }

    ImGui::PopStyleVar(2);

#if IMGUI_VERSION_NUM >= 17301
    ImGui::GetCurrentWindow()->ContentRegionRect.Max.x += frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->WorkRect.Max.x += frameHeight * 0.5f;
    ImGui::GetCurrentWindow()->InnerRect.Max.x += frameHeight * 0.5f;
#else
    ImGui::GetCurrentWindow()->ContentsRegionRect.Max.x += frameHeight * 0.5f;
#endif
    ImGui::GetCurrentWindow()->Size.x += frameHeight;

    ImGui::Dummy(ImVec2(0.0f, 0.0f));

    ImGui::EndGroup();
}
#endif

std::vector<float> animation::GetFloats(std::string vector_string) {
    std::vector<float> ret;
    auto stream = std::stringstream(vector_string);
    std::string token;
    while (std::getline(stream, token, ';')) {
        ret.push_back(std::stof(token));
    }
    return ret;
}

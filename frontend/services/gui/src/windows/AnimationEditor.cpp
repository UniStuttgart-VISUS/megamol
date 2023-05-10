/*
 * AnimationEditor.cpp
 *
 * Copyright (C) 2022 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "AnimationEditor.h"

#include "mmcore/param/EnumParam.h"
#include "mmcore/param/FlexEnumParam.h"
#include "mmcore/param/FloatParam.h"
#include "mmcore/param/IntParam.h"
#include "mmcore/param/StringParam.h"
#include "mmcore/param/Vector2fParam.h"
#include "mmcore/param/Vector3fParam.h"
#include "mmcore/param/Vector4fParam.h"
#include "mmcore/utility/animation/AnimationUtils.h"

#include "imgui_stdlib.h"
#include "nlohmann/json.hpp"

#include <cmath>
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
            accumulated_ms = fmodf(accumulated_ms, targeted_frame_time);
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
            if (frameskip > 0) {
                WriteValuesToGraph();
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
            if (found_idx == -1) {
                animation::FloatAnimation f = {the_name};
                allAnimations.emplace_back(f);
                found_idx = static_cast<int>(allAnimations.size()) - 1;
            }
            auto& a = std::get<animation::FloatAnimation>(allAnimations[found_idx]);
            if (a.HasKey(current_frame)) {
                a[current_frame].value = the_val;
            } else {
                a.AddKey({current_frame, the_val});
            }
            selectedStringKey = nullptr;
            if (found_idx == selectedAnimation) {
                selectedFloatKey = &a[current_frame];
            }
        }
        if (param_slot->Param<EnumParam>() || param_slot->Param<FlexEnumParam>() || param_slot->Param<StringParam>() ||
            param_slot->Param<FilePathParam>()) {
            if (found_idx == -1) {
                animation::StringAnimation s = {the_name};
                allAnimations.emplace_back(s);
                found_idx = static_cast<int>(allAnimations.size()) - 1;
            }
            auto& a = std::get<animation::StringAnimation>(allAnimations[found_idx]);
            if (a.HasKey(current_frame)) {
                a[current_frame].value = new_value;
            } else {
                a.AddKey({current_frame, new_value});
            }
            selectedFloatKey = nullptr;
            if (found_idx == selectedAnimation) {
                selectedStringKey = &a[current_frame];
            }
        }
        if (param_slot->Param<Vector2fParam>() || param_slot->Param<Vector3fParam>() ||
            param_slot->Param<Vector4fParam>()) {
            const auto vec = animation::GetFloats(new_value);
            if (found_idx == -1) {
                animation::FloatVectorAnimation v{the_name};
                allAnimations.emplace_back(v);
                found_idx = static_cast<int>(allAnimations.size()) - 1;
            }
            auto& a = std::get<animation::FloatVectorAnimation>(allAnimations[found_idx]);
            if (a.HasKey(current_frame)) {
                auto& k = a[current_frame];
                for (int i = 0; i < vec.size(); ++i) {
                    k.nestedData[i].value = vec[i];
                }
            } else {
                animation::VectorKey<animation::FloatKey> k;
                k.nestedData.resize(vec.size());
                for (int i = 0; i < vec.size(); ++i) {
                    k.nestedData[i].value = vec[i];
                    k.nestedData[i].time = current_frame;
                }
                a.AddKey(k);
            }
            selectedStringKey = nullptr;
            if (found_idx == selectedAnimation) {
                selectedFloatKey = &a[current_frame].nestedData.data()[0];
            }
        }
        if (found_idx == selectedAnimation) {
            CenterAnimation(allAnimations[found_idx]);
        }
    }
    // TODO: what else. Enum probably.
    return true;
}


void AnimationEditor::SpecificStateFromJSON(const nlohmann::json& in_json_all) {
    if (in_json_all.contains("AnimationEditor_State")) {
        auto& in_json = in_json_all["AnimationEditor_State"];
        in_json["write_to_graph"].get_to(write_to_graph);
        in_json["playing"].get_to(playing);
        in_json["playback_fps"].get_to(playback_fps);
        targeted_frame_time = 1000.0f / static_cast<float>(playback_fps);
        in_json["animation_bounds"].get_to(animation_bounds);
        in_json["current_frame"].get_to(current_frame);
        in_json["output_prefix"].get_to(output_prefix);
        if (in_json.contains("animation_file")) {
            in_json["animation_file"].get_to(animation_file);
        }

        animation::FloatAnimation f_dummy("dummy");
        animation::StringAnimation s_dummy("dummy");
        animation::FloatVectorAnimation v_dummy("dummy");
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
            } else if (j["type"] == "float_vector") {
                from_json(j, v_dummy);
                allAnimations.emplace_back(v_dummy);
            } else {
                error_popup_message = "Loading failed: " + j["name"].get<std::string>() + " has no known type";
                open_popup_error = true;
            }
        }
    }
}


void AnimationEditor::SpecificStateToJSON(nlohmann::json& inout_json_all) {
    auto& inout_json = inout_json_all["AnimationEditor_State"];
    inout_json["write_to_graph"] = write_to_graph;
    inout_json["playing"] = playing;
    inout_json["playback_fps"] = playback_fps;
    inout_json["animation_bounds"] = animation_bounds;
    inout_json["current_frame"] = current_frame;
    inout_json["animation_file"] = animation_file;
    inout_json["output_prefix"] = output_prefix;

    nlohmann::json anims;
    for (auto& fa : allAnimations) {
        std::visit([&](auto&& arg) -> void { anims.push_back(arg); }, fa);
    }
    inout_json["animations"] = anims;
}


void AnimationEditor::SetLastFrameMillis(float last_frame_millis) {
    last_frame_ms = last_frame_millis;
}

void AnimationEditor::RenderAnimation() {
    if (rendering) {
        // make screenshot and such
        std::stringstream command;
        if (current_frame >= animation_bounds[0] && current_frame <= animation_bounds[1]) {
            command << "mmScreenshot(\"" << output_prefix << "_" << std::setw(5) << std::setfill('0') << current_frame
                    << ".png\")\n";
            // that is without UI but without frame lag?
            //command << "mmScreenshotEntryPoint(\"\", \"" << output_prefix << "_" << std::setw(5) << std::setfill('0') << current_frame
            //        << ".png\")\n";
        }
        auto res = (*input_lua_func)(command.str());
        if (!std::get<0>(res)) {
            open_popup_error = true;
            error_popup_message = std::get<1>(res);
            playing = 0;
        }
        if (current_frame < animation_bounds[1]) {
            current_frame++;
            // this will set the state for the NEXT frame.
            // but we can grab only the front buffer, so we might need to wait one more frame. argh.
            (*input_lua_func)(GenerateLuaForFrame(current_frame));
        } else {
            rendering = false;
            auto res = (*input_lua_func)("mmExecCommand(\"_hotkey_gui_window_Animation Editor\")");
        }
    }
}


std::string AnimationEditor::GenerateLuaForFrame(animation::KeyTimeType time) {
    std::stringstream lua_commands;
    for (auto a : allAnimations) {
        std::visit(
            [&](auto&& arg) -> void {
                lua_commands << "mmSetParamValue(\"" << arg.GetName() << "\",[=[" << arg.GetValue(time) << "]=])\n";
            },
            a);
    }
    return lua_commands.str();
}

void AnimationEditor::WriteValuesToGraph() {
    if (write_to_graph) {
        const auto lua_commands = GenerateLuaForFrame(current_frame);
        const auto res = (*input_lua_func)(lua_commands);
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
    animation_file = file;
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
    animation_file = file;
    return true;
}


bool AnimationEditor::AnyParamHasKey(animation::KeyTimeType key_time) {
    bool has_key = false;
    for (auto& a : allAnimations) {
        std::visit([&](auto&& arg) -> void { has_key = has_key || arg.HasKey(key_time); }, a);
    }
    return has_key;
}


bool AnimationEditor::ExportToFile(const std::string& export_file) {
    std::ofstream out(export_file);
    if (!out.is_open()) {
        return false;
    }
    for (auto i = animation_bounds[0]; i <= animation_bounds[1]; ++i) {
        out << GenerateLuaForFrame(i);
        out << "mmRenderNextFrame()\n";
        switch (export_screenshot_option) {
        case 0: // never
            break;
        case 1: // every frame
            out << "mmScreenshot()\n";
            break;
        case 2: // if key
            if (AnyParamHasKey(i)) {
                out << "mmScreenshot()\n";
            }
        }
    }
    return true;
}


void AnimationEditor::DrawPopups() {
    if (this->file_browser.PopUp_Load(
            "Load Animation", animation_file, open_popup_load, {"anim"}, FilePathParam::Flag_File_RestrictExtension)) {
        LoadFromFile(animation_file);
        open_popup_load = false;
    }
    if (this->file_browser.PopUp_Save("Save Animation", animation_file, open_popup_save, {"anim"},
            FilePathParam::Flag_File_ToBeCreatedWithRestrExts, save_state, save_all_params)) {
        if (!SaveToFile(animation_file)) {
            error_popup_message = "Saving failed.";
            open_popup_error = true;
        }
        open_popup_save = false;
    }
    if (this->file_browser.PopUp_Save("Export Lua Script", export_file, open_popup_export, {"lua"},
            FilePathParam::Flag_File_ToBeCreatedWithRestrExts, save_state, save_all_params)) {
        if (!ExportToFile(export_file)) {
            error_popup_message = "Exporting failed.";
            open_popup_error = true;
        }
        open_popup_export = false;
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
            ImGui::Separator();
            if (ImGui::MenuItem("Load...")) {
                open_popup_load = true;
            }
            if (animation_file.empty()) {
                ImGui::BeginDisabled();
            }
            if (ImGui::MenuItem("Save")) {
                if (!SaveToFile(animation_file)) {
                    error_popup_message = "Saving failed.";
                    open_popup_error = true;
                }
            }
            if (animation_file.empty()) {
                ImGui::EndDisabled();
            }
            if (ImGui::MenuItem("Save as...")) {
                open_popup_save = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Export Lua...")) {
                open_popup_export = true;
            }
            const char* items[] = {"Never", "Every Frame", "On Key"};
            if (ImGui::BeginCombo("Screenshot", items[export_screenshot_option])) {
                for (int n = 0; n < 3; n++) {
                    const bool is_selected = (export_screenshot_option == n);
                    if (ImGui::Selectable(items[n], is_selected)) {
                        export_screenshot_option = n;
                    }
                }
                ImGui::EndCombo();
            }
            ImGui::EndMenu();
        }
        ImGui::Text(animation_file.c_str());
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
        WriteValuesToGraph();
    }
    ImGui::SameLine();
    DrawVerticalSeparator();

    ImGui::SameLine();
    if (ImGui::Button("delete animation")) {
        if (selectedAnimation != -1) {
            if (std::holds_alternative<animation::FloatVectorAnimation>(allAnimations[selectedAnimation])) {
                const auto& anim = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
                if (animEditorData.orientation_animation == &anim) {
                    animEditorData.orientation_animation = nullptr;
                }
                if (animEditorData.pos_animation == &anim) {
                    animEditorData.pos_animation = nullptr;
                }
            }
            allAnimations.erase(allAnimations.begin() + selectedAnimation);
            selectedFloatKey = nullptr;
            selectedStringKey = nullptr;
            selectedAnimation = -1;
        }
    }
    ImGui::SameLine();
    DrawVerticalSeparator();

    // keys
    ImGui::SameLine();
    if (ImGui::Button("add key")) {
        if (selectedAnimation != -1) {
            if (std::holds_alternative<animation::FloatAnimation>(allAnimations[selectedAnimation])) {
                auto& anim = std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]);
                if (!anim.HasKey(current_frame)) {
                    animation::FloatKey f;
                    f.time = current_frame;
                    f.value = anim.GetValue(current_frame);
                    f.interpolation = anim.GetInterpolation(current_frame);
                    // TODO: compute sensible tangent
                    auto t = anim.GetTangent(current_frame);
                    f.out_tangent = t;
                    f.in_tangent = ImVec2(-f.out_tangent.x, -f.out_tangent.y);
                    anim.AddKey(f);
                }
            } else if (std::holds_alternative<animation::FloatVectorAnimation>(allAnimations[selectedAnimation])) {
                auto& anim = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
                if (!anim.HasKey(current_frame)) {
                    animation::FloatVectorAnimation::KeyType v;
                    auto val_temp = anim.GetValue(current_frame);
                    auto int_temp = anim.GetInterpolation(current_frame);
                    auto tan_temp = anim.GetTangent(current_frame);
                    v.nestedData.resize(val_temp.size());
                    for (int i = 0; i < val_temp.size(); ++i) {
                        v.nestedData[i].time = current_frame;
                        v.nestedData[i].value = val_temp[i];
                        v.nestedData[i].interpolation = int_temp[i];
                        // TODO: compute sensible tangent
                        v.nestedData[i].out_tangent = tan_temp[i];
                        v.nestedData[i].in_tangent =
                            ImVec2(-v.nestedData[i].out_tangent.x, -v.nestedData[i].out_tangent.y);
                    }
                    anim.AddKey(v);
                }
            } else if (std::holds_alternative<animation::StringAnimation>(allAnimations[selectedAnimation])) {
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
    if (ImGui::Button("delete key")) {
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
            if (std::holds_alternative<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]) &&
                selectedFloatKey != nullptr) {
                std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation])
                    .DeleteKey(selectedFloatKey->time);
                selectedFloatKey = nullptr;
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
    } else if (std::holds_alternative<animation::FloatVectorAnimation>(anim)) {
        auto& a = std::get<animation::FloatVectorAnimation>(anim);
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
        if (std::fabs(max_val - min_val) < 0.1f) {
            max_val *= 1.0f;
            min_val -= 1.0f;
        }
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
    //selectedVec3Key = nullptr;
}

void AnimationEditor::DrawParams() {
    ImGui::Text("Available Parameters");
    bool have_pos = false, have_orient = false;
    if (selectedAnimation > -1 &&
        std::holds_alternative<animation::FloatVectorAnimation>(allAnimations[selectedAnimation])) {
        const auto& fva = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
        have_pos = fva.VectorLength() == 3;
        have_orient = fva.VectorLength() == 4;
    }
    if (!have_pos) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("use as 3D position")) {
        auto& fva = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
        animEditorData.pos_animation = &fva;
        animEditorData.active_region.first = animation_bounds[0];
        animEditorData.active_region.second = animation_bounds[1];
    }
    if (!have_pos) {
        ImGui::EndDisabled();
    }
    ImGui::SameLine();
    if (!have_orient) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("use as orientation")) {
        auto& fva = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
        animEditorData.orientation_animation = &fva;
        animEditorData.active_region.first = animation_bounds[0];
        animEditorData.active_region.second = animation_bounds[1];
    }
    if (!have_orient) {
        ImGui::EndDisabled();
    }
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
        if (ImGui::IsItemActivated()) {
            SelectAnimation(a);
            if (canvas_visible) {
                CenterAnimation(allAnimations[selectedAnimation]);
            }
        }
    }
    ImGui::EndChild();

    //animation::BeginGroupPanel("Animation", ImVec2(0.0f, 0.0f));
    ImGui::Text("Animation");
    ImGui::BeginChild(
        "Animation_SubWin", ImVec2(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y / 2.5f), true);
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
        animEditorData.active_region.first = animation_bounds[0];
        animEditorData.active_region.second = animation_bounds[1];
    }
    if (ImGui::SliderInt("Current Frame", &current_frame, animation_bounds[0], animation_bounds[1])) {
        WriteValuesToGraph();
    }
    ImGui::InputText("Output prefix", &output_prefix);
    if (output_prefix.empty()) {
        ImGui::BeginDisabled();
    }
    if (ImGui::Button("render")) {
        current_frame = animation_bounds[0] - 10; // a bit of pre-roll to let the GUI state settle
        auto res = (*input_lua_func)("mmExecCommand(\"_hotkey_gui_window_Animation Editor\")");
        write_to_graph = false;
        rendering = true;
    }
    if (output_prefix.empty()) {
        ImGui::EndDisabled();
    }
    //ImGui::Dummy(ImVec2(0.0f, 2.0f));
    //animation::EndGroupPanel();
    ImGui::EndChild();
}


void AnimationEditor::DrawInterpolation(
    ImDrawList* dl, const animation::FloatKey& key, const animation::FloatKey& key2, ImU32 line_col) {
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
            auto v1 = animation::FloatKey::InterpolateForParameter(key, key2, f);
            v1.y *= -1.0f;
            auto v2 = animation::FloatKey::InterpolateForParameter(key, key2, f + step);
            v2.y *= -1.0f;
            drawList->AddLine(v1 * custom_zoom, v2 * custom_zoom, reference_col);
        }
        for (auto t = key.time; t < key2.time; ++t) {
            auto v1 = animation::FloatKey::InterpolateForTime(key, key2, t);
            auto v2 = animation::FloatKey::InterpolateForTime(key, key2, t + 1);
            drawList->AddLine(ImVec2(t, v1 * -1.0f) * custom_zoom, ImVec2(t + 1, v2 * -1.0f) * custom_zoom, line_col);
        }
        break;
    }
    case animation::InterpolationType::SLERP: {
        // should never happen
        assert(false);
        break;
    }
    default:;
    }
}


void AnimationEditor::DrawFloatKey(
    ImDrawList* dl, animation::FloatKey& key, ImU32 col, animation::VectorKey<animation::FloatKey>* parent) {
    const float size = 4.0f;
    const ImVec2 button_size = {8.0f, 8.0f};
    auto active_key_color = IM_COL32(255, 192, 96, 255);
    auto tangent_color = IM_COL32(255, 255, 0, 255);

    auto drawList = ImGui::GetWindowDrawList();

    auto time = key.time;
    auto pos = ImVec2(time, key.value * -1.0f) * custom_zoom;
    auto t_out = ImVec2(time + key.out_tangent.x, (key.value + key.out_tangent.y) * -1.0f) * custom_zoom;

    const auto t_in = ImVec2(time + key.in_tangent.x, (key.value + key.in_tangent.y) * -1.0f) * custom_zoom;
    ImGui::SetCursorScreenPos(ImVec2{t_in.x - (button_size.x / 2.0f), t_in.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton(
        (std::string("##key_intan") + std::to_string(key.time) + std::to_string(col)).c_str(), button_size);
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
    ImGui::InvisibleButton(
        (std::string("##key_outtan") + std::to_string(key.time) + std::to_string(col)).c_str(), button_size);
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
        drawList->AddCircle(pos, size * 2.0f, col);
    }
    drawList->AddCircleFilled(pos, size, col);

    ImGui::SetCursorScreenPos(ImVec2{pos.x - (button_size.x / 2.0f), pos.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton(
        (std::string("##key") + std::to_string(key.time) + std::to_string(col)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        selectedFloatKey = &key;
        current_parent = parent;
        drag_start_value = selectedFloatKey->value;
        drag_start_time = selectedFloatKey->time;
        curr_interaction = InteractionType::DraggingKey;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey &&
        &key == selectedFloatKey) {
        const auto delta = ImGui::GetMouseDragDelta(ImGuiMouseButton_Left);
        selectedFloatKey->value = drag_start_value - delta.y / custom_zoom.y;
        auto new_time = drag_start_time + static_cast<animation::KeyTimeType>(delta.x / custom_zoom.x);
        if (new_time != selectedFloatKey->time) {
            if (parent == nullptr) {
                auto& anim = std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]);
                if (anim.HasKey(new_time)) {
                    // space is occupied. we do not want that.
                } else {
                    selectedFloatKey->time = new_time;
                }
            } else {
                auto& anim = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
                if (anim.HasKey(new_time)) {
                    // space is occupied. we do not want that.
                } else {
                    for (auto& fk : parent->nestedData) {
                        fk.time = new_time;
                    }
                }
            }
            WriteValuesToGraph();
        }
    }
    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey &&
        &key == selectedFloatKey) {
        if (drag_start_time != selectedFloatKey->time) {
            // fix sorting
            if (parent == nullptr) {
                auto& anim = std::get<animation::FloatAnimation>(allAnimations[selectedAnimation]);
                anim.FixSorting();
            } else {
                auto& anim = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
                anim.FixSorting();
            }
        }
        WriteValuesToGraph();
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


void AnimationEditor::DrawStringKey(ImDrawList* im_draws, animation::StringKey& key, ImU32 col) {
    const float size = 4.0f;
    const ImVec2 button_size = {8.0f, 8.0f};
    auto active_key_color = IM_COL32(255, 192, 96, 255);
    auto tangent_color = IM_COL32(255, 255, 0, 255);

    auto drawList = ImGui::GetWindowDrawList();

    auto time = key.time;
    auto pos = ImVec2(time, 0.0f) * custom_zoom;

    drawList->AddText(pos + button_size, col, key.value.c_str());
    if (selectedStringKey == &key) {
        drawList->AddCircle(pos, size * 2.0f, col);
    }
    drawList->AddCircleFilled(pos, size, col);

    ImGui::SetCursorScreenPos(ImVec2{pos.x - (button_size.x / 2.0f), pos.y - (button_size.y / 2.0f)});
    ImGui::InvisibleButton((std::string("##key") + std::to_string(key.time)).c_str(), button_size);
    if (ImGui::IsItemActivated()) {
        selectedStringKey = &key;
        drag_start_time = selectedStringKey->time;
        curr_interaction = InteractionType::DraggingKey;
    }
    if (ImGui::IsMouseDragging(ImGuiMouseButton_Left) && curr_interaction == InteractionType::DraggingKey &&
        &key == selectedStringKey) {
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
            } else if (std::holds_alternative<animation::FloatVectorAnimation>(allAnimations[selectedAnimation])) {
                auto& anim = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
                if (anim.GetSize() > 0) {
                    auto keys = anim.GetAllKeys();
                    const ImU32 cols[] = {IM_COL32(255, 0, 0, 255), IM_COL32(0, 255, 0, 255), IM_COL32(0, 0, 255, 255),
                        IM_COL32(255, 255, 255, 255)};
                    for (auto i = 0; i < keys.size(); ++i) {
                        auto& k = anim[keys[i]];
                        if (i < keys.size() - 1) {
                            auto& k2 = anim[keys[i + 1]];
                            if (k.nestedData[0].interpolation == animation::InterpolationType::SLERP) {
                                glm::quat q1(k.nestedData[3].value, k.nestedData[0].value, k.nestedData[1].value,
                                    k.nestedData[2].value);
                                glm::quat q2(k2.nestedData[3].value, k2.nestedData[0].value, k2.nestedData[1].value,
                                    k2.nestedData[2].value);
                                // fix "flipping" orientations while we are at it
                                if (glm::dot(q1, q2) < 0.0f) {
                                    q2 = -q2;
                                    k2.nestedData[0].value = q2.x;
                                    k2.nestedData[1].value = q2.y;
                                    k2.nestedData[2].value = q2.z;
                                    k2.nestedData[3].value = q2.w;
                                }
                                for (auto t = k.nestedData[0].time; t < k2.nestedData[0].time; ++t) {
                                    auto frac_t = (t - k.nestedData[0].time) /
                                                  static_cast<float>(k2.nestedData[0].time - k.nestedData[0].time);
                                    auto frac_t1 = (t + 1 - k.nestedData[0].time) /
                                                   static_cast<float>(k2.nestedData[0].time - k.nestedData[0].time);
                                    auto qi1 = glm::slerp(q1, q2, frac_t);
                                    auto qi2 = glm::slerp(q1, q2, frac_t1);
                                    for (int i = 0; i < 4; ++i) {
                                        drawList->AddLine(ImVec2(t, qi1[i] * -1.0f) * custom_zoom,
                                            ImVec2(t + 1, qi2[i] * -1.0f) * custom_zoom, cols[i]);
                                    }
                                }
                            } else {
                                for (int j = 0; j < k.nestedData.size(); ++j) {
                                    DrawInterpolation(drawList, k.nestedData[j], k2.nestedData[j], cols[j]);
                                }
                            }
                        }
                        for (int j = 0; j < k.nestedData.size(); ++j) {
                            DrawFloatKey(drawList, k.nestedData[j], cols[j], &k);
                        }
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
            custom_zoom.x = std::clamp(custom_zoom.x, 0.0001f, 4096.0f);
            custom_zoom.y = std::clamp(custom_zoom.y, 0.0001f, 4096.0f);
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
        auto& anim = allAnimations[selectedAnimation];
        if ((std::holds_alternative<animation::FloatAnimation>(anim) ||
                std::holds_alternative<animation::FloatVectorAnimation>(anim)) &&
            selectedFloatKey != nullptr) {
            auto old_time = selectedFloatKey->time;
            if (ImGui::InputInt("Time", &selectedFloatKey->time)) {
                // we must not move on top of another key -> map clash
                auto occupied =
                    std::visit([&](auto&& arg) -> bool { return arg.HasKey(selectedFloatKey->time); }, anim);
                if (occupied) {
                    selectedFloatKey->time = old_time;
                } else {
                    if (current_parent != nullptr) {
                        auto t = selectedFloatKey->time;
                        auto float_anim = std::get<animation::FloatVectorAnimation>(allAnimations[selectedAnimation]);
                        for (int i = 0; i < float_anim.VectorLength(); ++i) {
                            current_parent->nestedData[i].time = t;
                        }
                    }
                    std::visit([&](auto&& arg) -> void { arg.FixSorting(); }, anim);
                }
            }
            ImGui::InputFloat("Value", &selectedFloatKey->value);
            const char* items[] = {"Step", "Linear", "Hermite", "Cubic Bezier", "SLERP"};
            auto current_item = items[static_cast<int32_t>(selectedFloatKey->interpolation)];
            bool do_propagate = false;
            if (ImGui::BeginCombo("Interpolation", current_item)) {
                auto max_interp = current_parent != nullptr && current_parent->nestedData.size() == 4 ? 5 : 4;
                for (int n = 0; n < max_interp; n++) {
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
                        case 4:
                            selectedFloatKey->interpolation = animation::InterpolationType::SLERP;
                        }
                        do_propagate = true;
                    }
                    if (is_selected)
                        ImGui::SetItemDefaultFocus();
                }
                ImGui::EndCombo();
            }
            if (current_parent != nullptr) {
                if (ImGui::Checkbox("Propagate interpolation to siblings", &propagate_to_siblings)) {
                    do_propagate = true;
                }
                // SLERP cannot be selected for a single dimension.
                if ((propagate_to_siblings || selectedFloatKey->interpolation == animation::InterpolationType::SLERP) &&
                    do_propagate) {
                    auto interp = selectedFloatKey->interpolation;
                    for (int i = 0; i < current_parent->nestedData.size(); ++i) {
                        current_parent->nestedData[i].interpolation = interp;
                    }
                }
            }
            if (ImGui::Button("Set on all keys")) {
                if (std::holds_alternative<animation::FloatAnimation>(anim)) {
                    auto& fa = std::get<animation::FloatAnimation>(anim);
                    for (auto& k : fa) {
                        k.second.interpolation = selectedFloatKey->interpolation;
                    }
                } else if (std::holds_alternative<animation::VectorAnimation<animation::FloatKey>>(anim)) {
                    auto& va = std::get<animation::VectorAnimation<animation::FloatKey>>(anim);
                    for (auto& k : va) {
                        for (int i = 0; i < k.second.nestedData.size(); ++i) {
                            k.second.nestedData[i].interpolation = selectedFloatKey->interpolation;
                        }
                    }
                }
            }
        } else if (std::holds_alternative<animation::StringAnimation>(anim) && selectedStringKey != nullptr) {
            auto old_time = selectedStringKey->time;
            if (ImGui::InputInt("Time", &selectedStringKey->time)) {
                // we must not move on top of another key -> map clash
                auto occupied =
                    std::visit([&](auto&& arg) -> bool { return arg.HasKey(selectedStringKey->time); }, anim);
                if (occupied) {
                    selectedStringKey->time = old_time;
                } else {
                    std::visit([&](auto&& arg) -> void { arg.FixSorting(); }, anim);
                }
            }
            ImGui::InputText("Value", &selectedStringKey->value);
        }
    }
    ImGui::EndChild();
}


void AnimationEditor::DrawVerticalSeparator() {
    ImGui::BeginDisabled();
    ImGui::Button(" ", ImVec2(2, 0));
    ImGui::EndDisabled();
}


void AnimationEditor::DrawGrid() {
    auto drawList = ImGui::GetWindowDrawList();
    const auto top = canvas.ToLocal(ImVec2(0.0f, 0.0f));
    const auto bottom = canvas.ToLocal(canvas.Rect().GetBR());
    auto minorColor = ImGui::GetColorU32(ImGuiCol_Border);
    auto majorColor = ImGui::GetColorU32(ImGuiCol_Text);
    auto textColor = ImGui::GetColorU32(ImGuiCol_Text);

    auto region_start = ImVec2(-canvas.ViewOrigin().x, -canvas.Rect().GetHeight() + canvas.ViewOrigin().y);
    auto region_end = ImVec2(canvas.Rect().GetWidth() - canvas.ViewOrigin().x, canvas.ViewOrigin().y);
    region_start /= custom_zoom;
    region_end /= custom_zoom;
    auto width = region_end.x - region_start.x;

    auto dist_order = round(std::log10(width));
    auto majorUnit = powf(10, dist_order - 1.0f);
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

    auto region_start = ImVec2(-canvas.ViewOrigin().x, -canvas.Rect().GetHeight() + canvas.ViewOrigin().y);
    auto region_end = ImVec2(canvas.Rect().GetWidth() - canvas.ViewOrigin().x, canvas.ViewOrigin().y);
    region_start /= custom_zoom;
    region_end /= custom_zoom;
    auto height = region_end.y - region_start.y;

    auto dist_order = round(std::log10(height));
    auto majorUnit = powf(10, dist_order - 1.0f);
    auto minorUnit = majorUnit / 10.0f;

    auto minor_start = std::nearbyintf(region_start.y / minorUnit) * minorUnit;
    auto ref_pt = ImVec2(std::max(region_start.x, 0.0f * custom_zoom.x), 0.0f);
    ref_pt.x = std::min(ref_pt.x, region_end.x);
    for (auto s = minor_start; s < region_end.y; s += minorUnit) {
        drawList->AddLine(ref_pt * custom_zoom + ImVec2(-minorSize, -s * custom_zoom.y),
            ref_pt * custom_zoom + ImVec2(minorSize, -s * custom_zoom.y), minorColor);
    }

    auto major_start = std::nearbyintf(region_start.y / majorUnit) * majorUnit;
    for (auto s = major_start; s < region_end.y; s += majorUnit) {
        drawList->AddLine(ref_pt * custom_zoom + ImVec2(-minorSize, -s * custom_zoom.y),
            ref_pt * custom_zoom + ImVec2(minorSize, -s * custom_zoom.y), majorColor);

        char label[16];
        snprintf(label, 15, "%g", s);
        auto labelSize = ImGui::CalcTextSize(label);
        float label_offset = ref_pt.x > (region_end.x + region_start.x) / 2.0f ? -labelSize.x - minorSize : minorSize;

        auto labelPosition = ImVec2(ref_pt * custom_zoom + ImVec2(label_offset, -s * custom_zoom.y));
        drawList->AddText(labelPosition, textColor, label);
    }
}

/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "PerformanceMonitor.h"
#include "gui_utils.h"
#include <iomanip>


using namespace megamol;
using namespace megamol::gui;


PerformanceMonitor::PerformanceMonitor(const std::string& window_name)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_PERFORMANCE)
        , win_show_options(false)
        , win_buffer_size(20)
        , win_refresh_rate(2.0f)
        , win_mode(TIMINGMODE_FPS)
        , win_current_delay(0.0f)
        , win_ms_values()
        , win_fps_values()
        , win_ms_max(1.0f)
        , win_fps_max(1.0f)
        , frame_id(0)
        , averaged_fps(0.0f)
        , averaged_ms(0.0f) {

    // Configure FPS/MS Window
    this->win_config.flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar |
                             ImGuiWindowFlags_NoDocking | ImGuiWindowFlags_NoNavInputs;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F7, core::view::Modifier::NONE);
}


bool PerformanceMonitor::Update() {

    ImGuiIO& io = ImGui::GetIO();
    this->win_current_delay += io.DeltaTime;
    if (this->win_refresh_rate > 0.0f) {
        if (this->win_current_delay >= (1.0f / this->win_refresh_rate)) {

            // function for updating fps or ms
            auto update_values = [](float current_value, float& max_value, std::vector<float>& values,
                                     size_t actual_buffer_size) {
                size_t buffer_size = values.size();
                if (buffer_size != actual_buffer_size) {
                    if (buffer_size > actual_buffer_size) {
                        values.erase(
                            values.begin(), values.begin() + static_cast<long>(buffer_size - actual_buffer_size));

                    } else if (buffer_size < actual_buffer_size) {
                        values.insert(values.begin(), (actual_buffer_size - buffer_size), 0.0f);
                    }
                }
                if (buffer_size > 0) {
                    values.erase(values.begin());
                    values.emplace_back(static_cast<float>(current_value));
                    float new_max_value = 0.0f;
                    for (auto& v : values) {
                        new_max_value = std::max(v, new_max_value);
                    }
                    max_value = new_max_value;
                }
            };

            update_values(((this->averaged_fps == 0.0f) ? (1.0f / io.DeltaTime) : (this->averaged_fps)),
                this->win_fps_max, this->win_fps_values, this->win_buffer_size);

            update_values(((this->averaged_ms == 0.0f) ? (io.DeltaTime * 1000.0f) : (this->averaged_ms)),
                this->win_ms_max, this->win_ms_values, this->win_buffer_size);

            this->win_current_delay = 0.0f;
        }
    }
    return true;
}


bool PerformanceMonitor::Draw() {

    ImGuiStyle& style = ImGui::GetStyle();

    if (ImGui::RadioButton("fps", (this->win_mode == TIMINGMODE_FPS))) {
        this->win_mode = TIMINGMODE_FPS;
    }
    ImGui::SameLine();

    if (ImGui::RadioButton("ms", (this->win_mode == TIMINGMODE_MS))) {
        this->win_mode = TIMINGMODE_MS;
    }

    ImGui::TextDisabled("Frame ID:");
    ImGui::SameLine();
    ImGui::Text("%lu", this->frame_id);

    ImGui::SameLine(
        ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() - style.ItemSpacing.x - style.ItemInnerSpacing.x));
    if (ImGui::ArrowButton("Options_", ((this->win_show_options) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
        this->win_show_options = !this->win_show_options;
    }

    auto* value_buffer = ((this->win_mode == TIMINGMODE_FPS) ? (&this->win_fps_values) : (&this->win_ms_values));
    int buffer_size = static_cast<int>(value_buffer->size());

    std::string value_string;
    if (buffer_size > 0) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << value_buffer->back();
        value_string = stream.str();
    }

    float* value_ptr = value_buffer->data();
    float max_value = ((this->win_mode == TIMINGMODE_FPS) ? (this->win_fps_max) : (this->win_ms_max));
    ImGui::PlotLines("###msplot", value_ptr, buffer_size, 0, value_string.c_str(), 0.0f, (1.5f * max_value),
        ImVec2(0.0f, (50.0f * megamol::gui::gui_scaling.Get())));

    if (this->win_show_options) {
        if (ImGui::InputFloat("Refresh Rate (per sec.)", &this->win_refresh_rate, 1.0f, 10.0f, "%.3f",
                ImGuiInputTextFlags_EnterReturnsTrue)) {
            this->win_refresh_rate = std::max(1.0f, this->win_refresh_rate);
        }

        if (ImGui::InputInt("History Size", &this->win_buffer_size, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) {
            this->win_buffer_size = std::max(1, this->win_buffer_size);
        }

        if (ImGui::Button("Current Value")) {
            ImGui::SetClipboardText(value_string.c_str());
        }
        ImGui::SameLine();

        if (ImGui::Button("All Values")) {
            std::stringstream stream;
            stream << std::fixed << std::setprecision(3);
            auto reverse_end = value_buffer->rend();
            for (auto i = value_buffer->rbegin(); i != reverse_end; ++i) {
                stream << (*i) << "\n";
            }
            ImGui::SetClipboardText(stream.str().c_str());
        }
        ImGui::SameLine();
        ImGui::TextUnformatted("Copy to Clipborad");
        std::string help("Values are listed in chronological order (newest first).");
        this->tooltip.Marker(help);
    }

    return true;
}


void PerformanceMonitor::SpecificStateFromJSON(const nlohmann::json& in_json) {

    for (auto& header_item : in_json.items()) {
        if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
            for (auto& config_item : header_item.value().items()) {
                if (config_item.key() == this->Name()) {
                    auto config_values = config_item.value();

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"fpsms_show_options"}, &this->win_show_options);
                    megamol::core::utility::get_json_value<int>(
                        config_values, {"fpsms_max_value_count"}, &this->win_buffer_size);
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"fpsms_refresh_rate"}, &this->win_refresh_rate);
                    int mode = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"fpsms_mode"}, &mode);
                    this->win_mode = static_cast<TimingMode>(mode);
                }
            }
        }
    }
}


void PerformanceMonitor::SpecificStateToJSON(nlohmann::json& inout_json) {

    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["fpsms_show_options"] = this->win_show_options;
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["fpsms_max_value_count"] = this->win_buffer_size;
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["fpsms_refresh_rate"] = this->win_refresh_rate;
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["fpsms_mode"] = static_cast<int>(this->win_mode);
}

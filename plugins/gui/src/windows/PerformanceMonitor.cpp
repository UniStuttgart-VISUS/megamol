/*
 * PerformanceMonitor.cpp
 *
 * Copyright (C) 2021 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "PerformanceMonitor.h"


using namespace megamol;
using namespace megamol::gui;

PerformanceMonitor::PerformanceMonitor() : WindowConfiguration("Performance Metrics", WindowConfiguration::WINDOW_ID_PERFORMANCE) {

    // Configure FPS/MS Window
    this->config.hotkey = core::view::KeyCode(core::view::Key::KEY_F7);
    this->config.flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoDocking;
}


void PerformanceMonitor::Update() {

    ImGuiIO& io = ImGui::GetIO();

    wc.config.specific.tmp_current_delay += io.DeltaTime;
    if (wc.config.specific.fpsms_refresh_rate > 0.0f) {
        if (wc.config.specific.tmp_current_delay >= (1.0f / wc.config.specific.fpsms_refresh_rate)) {

            auto update_values = [](float current_value, float& max_value, std::vector<float>& values,
                                    size_t actual_buffer_size) {
                size_t buffer_size = values.size();
                if (buffer_size != actual_buffer_size) {
                    if (buffer_size > actual_buffer_size) {
                        values.erase(values.begin(), values.begin() + (buffer_size - actual_buffer_size));

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

            update_values(
                    ((this->state.stat_averaged_fps == 0.0f) ? (1.0f / io.DeltaTime) : (this->state.stat_averaged_fps)),
                    wc.config.specific.tmp_fps_max, wc.config.specific.tmp_fps_values,
                    wc.config.specific.fpsms_buffer_size);

            update_values(
                    ((this->state.stat_averaged_ms == 0.0f) ? (io.DeltaTime * 1000.0f) : (this->state.stat_averaged_ms)),
                    wc.config.specific.tmp_ms_max, wc.config.specific.tmp_ms_values, wc.config.specific.fpsms_buffer_size);

            wc.config.specific.tmp_current_delay = 0.0f;
        }
    }
}


void PerformanceMonitor::Draw() {

    ImGuiStyle& style = ImGui::GetStyle();

    if (ImGui::RadioButton("fps", (wc.config.specific.fpsms_mode == WindowConfiguration::TIMINGMODE_FPS))) {
        wc.config.specific.fpsms_mode = WindowConfiguration::TIMINGMODE_FPS;
    }
    ImGui::SameLine();

    if (ImGui::RadioButton("ms", (wc.config.specific.fpsms_mode == WindowConfiguration::TIMINGMODE_MS))) {
        wc.config.specific.fpsms_mode = WindowConfiguration::TIMINGMODE_MS;
    }

    ImGui::TextDisabled("Frame ID:");
    ImGui::SameLine();
    auto frameid = this->state.stat_frame_count;
    ImGui::Text("%lu", frameid);

    ImGui::SameLine(
            ImGui::CalcItemWidth() - (ImGui::GetFrameHeightWithSpacing() - style.ItemSpacing.x - style.ItemInnerSpacing.x));
    if (ImGui::ArrowButton("Options_", ((wc.config.specific.fpsms_show_options) ? (ImGuiDir_Down) : (ImGuiDir_Up)))) {
        wc.config.specific.fpsms_show_options = !wc.config.specific.fpsms_show_options;
    }

    auto* value_buffer =
            ((wc.config.specific.fpsms_mode == WindowConfiguration::TIMINGMODE_FPS) ? (&wc.config.specific.tmp_fps_values)
                                                                                    : (&wc.config.specific.tmp_ms_values));
    int buffer_size = static_cast<int>(value_buffer->size());

    std::string value_string;
    if (buffer_size > 0) {
        std::stringstream stream;
        stream << std::fixed << std::setprecision(3) << value_buffer->back();
        value_string = stream.str();
    }

    float* value_ptr = value_buffer->data();
    float max_value =
            ((wc.config.specific.fpsms_mode == WindowConfiguration::TIMINGMODE_FPS) ? (wc.config.specific.tmp_fps_max)
                                                                                    : (wc.config.specific.tmp_ms_max));
    ImGui::PlotLines("###msplot", value_ptr, buffer_size, 0, value_string.c_str(), 0.0f, (1.5f * max_value),
                     ImVec2(0.0f, (50.0f * megamol::gui::gui_scaling.Get())));

    if (wc.config.specific.fpsms_show_options) {
        if (ImGui::InputFloat("Refresh Rate (per sec.)", &wc.config.specific.fpsms_refresh_rate, 1.0f, 10.0f, "%.3f",
                              ImGuiInputTextFlags_EnterReturnsTrue)) {
            wc.config.specific.fpsms_refresh_rate = std::max(1.0f, wc.config.specific.fpsms_refresh_rate);
        }

        if (ImGui::InputInt(
                "History Size", &wc.config.specific.fpsms_buffer_size, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) {
            wc.config.specific.fpsms_buffer_size = std::max(1, wc.config.specific.fpsms_buffer_size);
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
}

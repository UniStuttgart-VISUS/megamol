/*
 * WindowCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "AbstractWindow.h"
#include "gui_utils.h"


using namespace megamol;
using namespace megamol::gui;


void AbstractWindow::ApplyWindowSizePosition(bool consider_menu) {

    assert(ImGui::GetCurrentContext() != nullptr);

    ImGuiIO& io = ImGui::GetIO();

    // Main menu height
    float y_offset = ImGui::GetFrameHeight();

    ImVec2 win_pos = this->win_config.position;
    ImVec2 win_size = this->win_config.size;
    if (this->win_config.flags & ImGuiWindowFlags_AlwaysAutoResize) {
        win_size = ImGui::GetWindowSize();
    }

    // Fit max window size to viewport
    if (win_size.x > io.DisplaySize.x) {
        win_size.x = io.DisplaySize.x;
    }
    if (win_size.y > (io.DisplaySize.y - y_offset)) {
        win_size.y = (io.DisplaySize.y - y_offset);
    }

    // Snap to viewport
    /// ImGui automatically moves windows lying outside viewport
    // float win_width = io.DisplaySize.x - (win_pos.x);
    // if (win_width < win_size.x) {
    //    win_pos.x = io.DisplaySize.x - (win_size.x);
    //}
    // float win_height = io.DisplaySize.y - (win_pos.y);
    // if (win_height < win_size.y) {
    //    win_pos.y = io.DisplaySize.y - (win_size.y);
    //}
    // if (win_pos.x < 0) {
    //    win_pos.x = 0.0f;
    //}

    // Snap window below menu bar
    if (consider_menu && (win_pos.y < y_offset)) {
        win_pos.y = y_offset;
    }

    this->win_config.position = win_pos;
    // wc.config.reset_position = win_pos;
    ImGui::SetWindowPos(win_pos, ImGuiCond_Always);

    this->win_config.size = win_size;
    // wc.config.reset_size = win_size;
    ImGui::SetWindowSize(win_size, ImGuiCond_Always);
}


void AbstractWindow::WindowContextMenu(bool menu_visible, bool& out_collapsing_changed) {

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 viewport = io.DisplaySize;
    out_collapsing_changed = false;
    float y_offset = (menu_visible) ? (ImGui::GetFrameHeight()) : (0.0f);
    ImVec2 window_viewport = ImVec2(viewport.x, viewport.y - y_offset);
    bool window_maximized = (this->win_config.size == window_viewport);
    bool toggle_window_size = false; // (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left));

    // Context Menu
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem(((window_maximized) ? ("Minimize") : ("Maximize")))) {
            toggle_window_size = true;
        }
        if (ImGui::MenuItem(((!this->win_config.collapsed) ? ("Collapse") : ("Expand")), "Double Left Click")) {
            this->win_config.collapsed = !this->win_config.collapsed;
            out_collapsing_changed = true;
        }

        if (ImGui::MenuItem("Full Width", nullptr)) {
            this->win_config.size.x = viewport.x;
            this->win_config.reset_pos_size = true;
        }
        ImGui::Separator();

/// DOCKING
#ifdef IMGUI_HAS_DOCK
        ImGui::MenuItem("Docking", "Shift + Left-Drag", false, false);
        ImGui::Separator();
#endif
        ImGui::MenuItem("Snap", nullptr, false, false);

        if (ImGui::ArrowButton("snap_left", ImGuiDir_Left)) {
            this->win_config.position.x = 0.0f;
            this->win_config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("snap_up", ImGuiDir_Up)) {
            this->win_config.position.y = 0.0f;
            this->win_config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("snap_down", ImGuiDir_Down)) {
            this->win_config.position.y = viewport.y - this->win_config.size.y;
            this->win_config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("snap_right", ImGuiDir_Right)) {
            this->win_config.position.x = viewport.x - this->win_config.size.x;
            this->win_config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();

        if (ImGui::MenuItem("Close", nullptr)) {
            this->win_config.show = false;
        }
        ImGui::EndPopup();
    }

    // Toggle window size
    if (toggle_window_size) {
        if (window_maximized) {
            // Window is maximized
            this->win_config.size = this->win_config.reset_size;
            this->win_config.position = this->win_config.reset_position;
            this->win_config.reset_pos_size = true;
        } else {
            // Window is minimized
            window_viewport = ImVec2(viewport.x, viewport.y - y_offset);
            this->win_config.reset_size = this->win_config.size;
            this->win_config.reset_position = this->win_config.position;
            this->win_config.size = window_viewport;
            this->win_config.position = ImVec2(0.0f, y_offset);
            this->win_config.reset_pos_size = true;
        }
    }
}


void AbstractWindow::StateFromJSON(const nlohmann::json& in_json) {

    for (auto& header_item : in_json.items()) {
        if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
            for (auto& config_item : header_item.value().items()) {
                if (config_item.key() == this->Name()) {
                    auto config_values = config_item.value();

                    int win_flags = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"win_flags"}, &win_flags);
                    this->win_config.flags = static_cast<ImGuiWindowFlags>(win_flags);
                    megamol::core::utility::get_json_value<bool>(config_values, {"win_show"}, &this->win_config.show);
                    std::array<int, 2> hotkey = {0, 0};
                    megamol::core::utility::get_json_value<int>(
                        config_values, {"win_hotkey"}, hotkey.data(), hotkey.size());
                    this->win_config.hotkey = core::view::KeyCode(
                        static_cast<core::view::Key>(hotkey[0]), static_cast<core::view::Modifiers>(hotkey[1]));
                    std::array<float, 2> position = {0.0f, 0.0f};
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_position"}, position.data(), position.size());
                    this->win_config.position = ImVec2(position[0], position[1]);
                    std::array<float, 2> size = {0.0f, 0.0f};
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_size"}, size.data(), size.size());
                    this->win_config.size = ImVec2(size[0], size[1]);
                    std::array<float, 2> reset_size = {0.0f, 0.0f};
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_reset_size"}, reset_size.data(), reset_size.size());
                    this->win_config.reset_size = ImVec2(reset_size[0], reset_size[1]);
                    std::array<float, 2> reset_position = {0.0f, 0.0f};
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_reset_position"}, reset_position.data(), reset_position.size());
                    this->win_config.reset_position = ImVec2(reset_position[0], reset_position[1]);
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"win_collapsed"}, &this->win_config.collapsed);
                    this->win_config.reset_pos_size = true;
                }
            }
        }
    }
}


void AbstractWindow::StateToJSON(nlohmann::json& inout_json) {

    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_show"] = this->win_config.show;
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_flags"] = static_cast<int>(this->win_config.flags);
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_hotkey"] = {
        static_cast<int>(this->win_config.hotkey.key), this->win_config.hotkey.mods.toInt()};
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_position"] = {
        this->win_config.position.x, this->win_config.position.y};
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_size"] = {
        this->win_config.size.x, this->win_config.size.y};
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_reset_size"] = {
        this->win_config.reset_size.x, this->win_config.reset_size.y};
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_reset_position"] = {
        this->win_config.reset_position.x, this->win_config.reset_position.y};
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["win_collapsed"] = this->win_config.collapsed;
}

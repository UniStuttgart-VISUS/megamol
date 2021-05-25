/*
 * WindowCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "gui_utils.h"
#include "WindowConfiguration.h"


using namespace megamol;
using namespace megamol::gui;


void WindowConfiguration::ApplyWindowSizePosition(bool consider_menu) {

    assert(ImGui::GetCurrentContext() != nullptr);

    ImGuiIO& io = ImGui::GetIO();

    // Main menu height
    float y_offset = ImGui::GetFrameHeight();

    ImVec2 win_pos = this->config.position;
    ImVec2 win_size = this->config.size;
    if (this->config.flags & ImGuiWindowFlags_AlwaysAutoResize) {
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

    this->config.position = win_pos;
    // wc.config.reset_position = win_pos;
    ImGui::SetWindowPos(win_pos, ImGuiCond_Always);

    this->config.size = win_size;
    // wc.config.reset_size = win_size;
    ImGui::SetWindowSize(win_size, ImGuiCond_Always);
}


void WindowConfiguration::WindowContextMenu(bool menu_visible, bool& out_collapsing_changed) {

    ImGuiIO& io = ImGui::GetIO();
    ImVec2 viewport = io.DisplaySize;
    out_collapsing_changed = false;
    float y_offset = (menu_visible) ? (ImGui::GetFrameHeight()) : (0.0f);
    ImVec2 window_viewport = ImVec2(viewport.x, viewport.y - y_offset);
    bool window_maximized = (this->config.size == window_viewport);
    bool toggle_window_size = false; // (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(0));

    // Context Menu
    if (ImGui::BeginPopupContextItem()) {
        if (ImGui::MenuItem(((window_maximized) ? ("Minimize") : ("Maximize")))) {
            toggle_window_size = true;
        }
        if (ImGui::MenuItem(((!this->config.collapsed) ? ("Collapse") : ("Expand")), "Double Left Click")) {
            this->config.collapsed = !this->config.collapsed;
            out_collapsing_changed = true;
        }

        if (ImGui::MenuItem("Full Width", nullptr)) {
            this->config.size.x = viewport.x;
            this->config.reset_pos_size = true;
        }
        ImGui::Separator();

/// DOCKING
#ifdef IMGUI_HAS_DOCK
        ImGui::MenuItem("Docking", "Shift + Left-Drag", false, false);
        ImGui::Separator();
#endif
        ImGui::MenuItem("Snap", nullptr, false, false);

        if (ImGui::ArrowButton("snap_left", ImGuiDir_Left)) {
            this->config.position.x = 0.0f;
            this->config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("snap_up", ImGuiDir_Up)) {
            this->config.position.y = 0.0f;
            this->config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("snap_down", ImGuiDir_Down)) {
            this->config.position.y = viewport.y - this->config.size.y;
            this->config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::SameLine();
        if (ImGui::ArrowButton("snap_right", ImGuiDir_Right)) {
            this->config.position.x = viewport.x - this->config.size.x;
            this->config.reset_pos_size = true;
            ImGui::CloseCurrentPopup();
        }
        ImGui::Separator();

        if (ImGui::MenuItem("Close", nullptr)) {
            this->config.show = false;
        }
        ImGui::EndPopup();
    }

    // Toggle window size
    if (toggle_window_size) {
        if (window_maximized) {
            // Window is maximized
            this->config.size = this->config.reset_size;
            this->config.position = this->config.reset_position;
            this->config.reset_pos_size = true;
        } else {
            // Window is minimized
            window_viewport = ImVec2(viewport.x, viewport.y - y_offset);
            this->config.reset_size = this->config.size;
            this->config.reset_position = this->config.position;
            this->config.size = window_viewport;
            this->config.position = ImVec2(0.0f, y_offset);
            this->config.reset_pos_size = true;
        }
    }

    // Apply window position and size
    if (this->config.reset_pos_size || (menu_visible && ImGui::IsMouseReleased(0) &&
                                       ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows))) {
        this->ApplyWindowSizePosition(menu_visible);
        this->config.reset_pos_size = false;
    }
}


bool WindowConfiguration::StateFromJSON(const nlohmann::json &in_json) {


    return false;
}


bool WindowConfiguration::StateToJSON(nlohmann::json &inout_json) {


    return false;
}
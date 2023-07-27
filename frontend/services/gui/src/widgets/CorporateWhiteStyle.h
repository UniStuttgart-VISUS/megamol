/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once


#include "DefaultStyle.h"


/** ************************************************************************
 * A "CorporateWhite"-Style
 *
 * TODO Complete with more colors
 */
inline void CorporateWhiteStyle(int is3D = 0) {

    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

    // Geometry
    DefaultStyle();
    style.PopupRounding = 3.0f;
    style.WindowPadding = ImVec2(4.0f, 4.0f);
    style.FramePadding = ImVec2(6.0f, 4.0f);
    style.ItemSpacing = ImVec2(6.0f, 2.0f);
    style.ScrollbarSize = 18.0f;
    style.WindowBorderSize = 1.0f;
    style.ChildBorderSize = 1.0f;
    style.PopupBorderSize = 1.0f;
    style.FrameBorderSize = static_cast<float>(is3D);
    style.WindowRounding = 0.0f; // 3;
    style.ChildRounding = 3.0f;
    style.FrameRounding = 3.0f;
    style.ScrollbarRounding = 2.0f;
    style.GrabRounding = 3.0f;
    style.TabBorderSize = static_cast<float>(is3D);
    style.TabRounding = 3.0f;


    // Colors
    ImVec4 white = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    ImVec4 transparent = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    ImVec4 dark = ImVec4(0.00f, 0.00f, 0.00f, 0.20f);
    ImVec4 darker = ImVec4(0.00f, 0.00f, 0.00f, 0.50f);
    ImVec4 background = ImVec4(0.95f, 0.95f, 0.95f, 1.00f);
    ImVec4 text = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    ImVec4 border = ImVec4(0.60f, 0.60f, 0.60f, 1.00f);
    ImVec4 grab = ImVec4(0.69f, 0.69f, 0.69f, 1.00f);
    ImVec4 header = ImVec4(0.86f, 0.86f, 0.86f, 1.00f);
    ImVec4 active = ImVec4(0.00f, 0.47f, 0.84f, 1.00f);
    ImVec4 hover = ImVec4(0.00f, 0.47f, 0.84f, 0.20f);
    ImVec4 gray = ImVec4(0.75f, 0.75f, 0.75f, 0.20f);

    colors[ImGuiCol_Text] = text;
    colors[ImGuiCol_TextDisabled] = gray;
    colors[ImGuiCol_WindowBg] = background;
    colors[ImGuiCol_ChildBg] = background;
    colors[ImGuiCol_PopupBg] = white;
    colors[ImGuiCol_Border] = border;
    colors[ImGuiCol_BorderShadow] = transparent;
    colors[ImGuiCol_FrameBg] = white;
    colors[ImGuiCol_FrameBgHovered] = hover;
    colors[ImGuiCol_FrameBgActive] = active;
    colors[ImGuiCol_TitleBg] = header;
    colors[ImGuiCol_TitleBgActive] = active;
    colors[ImGuiCol_TitleBgCollapsed] = header;
    colors[ImGuiCol_CheckMark] = text;
    colors[ImGuiCol_SliderGrab] = grab;
    colors[ImGuiCol_SliderGrabActive] = darker;
    colors[ImGuiCol_Button] = header;
    colors[ImGuiCol_ButtonHovered] = hover;
    colors[ImGuiCol_ButtonActive] = active;
    colors[ImGuiCol_MenuBarBg] = header;
    colors[ImGuiCol_Header] = header;
    colors[ImGuiCol_HeaderHovered] = hover;
    colors[ImGuiCol_HeaderActive] = active;
    colors[ImGuiCol_ScrollbarBg] = header;
    colors[ImGuiCol_ScrollbarGrab] = grab;
    colors[ImGuiCol_ScrollbarGrabHovered] = dark;
    colors[ImGuiCol_ScrollbarGrabActive] = darker;
    colors[ImGuiCol_Separator] = ImVec4(0.000f, 0.000f, 0.000f, 0.137f);
    colors[ImGuiCol_SeparatorHovered] = ImVec4(0.700f, 0.671f, 0.600f, 0.290f);
    colors[ImGuiCol_SeparatorActive] = ImVec4(0.702f, 0.671f, 0.600f, 0.674f);
    colors[ImGuiCol_ResizeGrip] = ImVec4(0.87f, 0.87f, 0.87f, 0.53f);
    colors[ImGuiCol_ResizeGripHovered] = ImVec4(0.87f, 0.87f, 0.87f, 0.74f);
    colors[ImGuiCol_ResizeGripActive] = ImVec4(0.87f, 0.87f, 0.87f, 0.74f);
    colors[ImGuiCol_Tab] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_TabHovered] = ImVec4(0.40f, 0.40f, 0.40f, 1.00f);
    colors[ImGuiCol_TabActive] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
    colors[ImGuiCol_TabUnfocused] = ImVec4(0.25f, 0.25f, 0.25f, 1.00f);
    colors[ImGuiCol_TabUnfocusedActive] = ImVec4(0.33f, 0.33f, 0.33f, 1.00f);
    colors[ImGuiCol_PlotLines] = ImVec4(0.61f, 0.61f, 0.61f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered] = ImVec4(1.00f, 0.43f, 0.35f, 1.00f);
    colors[ImGuiCol_PlotHistogram] = ImVec4(0.90f, 0.70f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered] = ImVec4(1.00f, 0.60f, 0.00f, 1.00f);
    colors[ImGuiCol_TextSelectedBg] = ImVec4(0.73f, 0.73f, 0.73f, 0.35f);
    colors[ImGuiCol_ModalWindowDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.35f);
    colors[ImGuiCol_DragDropTarget] = ImVec4(1.00f, 1.00f, 0.00f, 0.90f);
    colors[ImGuiCol_NavHighlight] = ImVec4(0.26f, 0.59f, 0.98f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight] = ImVec4(1.00f, 1.00f, 1.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg] = ImVec4(0.80f, 0.80f, 0.80f, 0.20f);
#ifdef IMGUI_HAS_DOCK
    colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.85f, 0.85f, 0.85f, 0.28f);

    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
#endif
}

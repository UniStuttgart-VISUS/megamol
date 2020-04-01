/*
 * CorporateWhiteStyle.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_CORPORATEWHITESTYLE_INCLUDED
#define MEGAMOL_GUI_CORPORATEWHITESTYLE_INCLUDED

#include <imgui.h>


/**
 * A "CorporateWhite"-Style.
 *
 * TODO: complete with more colors
 */
inline void CorporateWhiteStyle(int is3D = 0) {
    ImGuiStyle& style = ImGui::GetStyle();
    ImVec4* colors = style.Colors;

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

    colors[ImGuiCol_Text] = text;
    colors[ImGuiCol_TextDisabled] = text;
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

    // colors[ImGuiCol_Separator] =
    // colors[ImGuiCol_SeparatorHovered] =
    // colors[ImGuiCol_SeparatorActive] =
    // colors[ImGuiCol_ResizeGrip] =
    // colors[ImGuiCol_ResizeGripHovered] =
    // colors[ImGuiCol_ResizeGripActive] =
    // colors[ImGuiCol_Tab] =
    // colors[ImGuiCol_TabHovered] =
    // colors[ImGuiCol_TabActive] =
    // colors[ImGuiCol_TabUnfocused] =
    // colors[ImGuiCol_TabUnfocusedActive] =
    // colors[ImGuiCol_PlotLines] =
    // colors[ImGuiCol_PlotLinesHovered] =
    // colors[ImGuiCol_PlotHistogram] =
    // colors[ImGuiCol_PlotHistogramHovered] =
    // colors[ImGuiCol_TextSelectedBg] =
    // colors[ImGuiCol_ModalWindowDimBg] =
    // colors[ImGuiCol_DragDropTarget] =
    // colors[ImGuiCol_NavHighlight] =
    // colors[ImGuiCol_NavWindowingHighlight] =
    // colors[ImGuiCol_NavWindowingDimBg] =

    style.PopupRounding = 3;

    style.WindowPadding = ImVec2(4, 4);
    style.FramePadding = ImVec2(6, 4);
    style.ItemSpacing = ImVec2(6, 2);

    style.ScrollbarSize = 18;

    style.WindowBorderSize = 1;
    style.ChildBorderSize = 1;
    style.PopupBorderSize = 1;
    style.FrameBorderSize = static_cast<float>(is3D);

    style.WindowRounding = 3;
    style.ChildRounding = 3;
    style.FrameRounding = 3;
    style.ScrollbarRounding = 2;
    style.GrabRounding = 3;

    style.TabBorderSize = static_cast<float>(is3D);
    style.TabRounding = 3;

#ifdef IMGUI_HAS_DOCK
    colors[ImGuiCol_DockingEmptyBg] = ImVec4(0.38f, 0.38f, 0.38f, 1.00f);
    colors[ImGuiCol_DockingPreview] = ImVec4(0.85f, 0.85f, 0.85f, 0.28f);

    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }
#endif
}

#endif // MEGAMOL_GUI_CORPORATEWHITESTYLE_INCLUDED
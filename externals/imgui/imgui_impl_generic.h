#pragma once
#include "imgui.h"
#include <array>

struct GenericWindow {
    int width;
    int height;
    int x;
    int y;
};

struct GenericMonitor {
    int res_x;
    int res_y;
};

struct GenericCursor {
    std::array<float, 2> pos;
};

IMGUI_IMPL_API bool ImGui_ImplGeneric_Init(GenericWindow* window);
IMGUI_IMPL_API void ImGui_ImplGeneric_Shutdown();
IMGUI_IMPL_API void ImGui_ImplGeneric_NewFrame(GenericWindow* window, GenericMonitor* monitor);

// ImGuiTexInspect, a texture inspector widget for dear imgui

#pragma once
#include "imgui.h"

namespace ImGuiTexInspect
{
struct Texture
{
    void *texture;
    ImVec2 size;
};

//Texture LoadTexture(const char *path);

void ShowDemoWindow();

void Demo_SetTexture(ImTextureID tex, ImVec2 size);
} //namespace ImGuiTexInspect

#pragma once

#include "imgui.h"
#include "imgui_tex_inspect.h"

namespace ImGuiTexInspect {
// Source: https://github.com/andyborrell/imgui_tex_inspect/blob/80ffc679e8f3f477d861d7a806e072098e94158c/imgui_tex_inspect_demo.h#L8-L12
struct Texture
{
    void* texture;
    ImVec2 size;
};

void Demo_ColorFilters(const Texture& testTex, InspectorFlags inFlags);
void Demo_ColorMatrix(const Texture& testTex, InspectorFlags inFlags);
void Demo_AlphaMode(const Texture& testTex, InspectorFlags inFlags);
void Demo_WrapAndFilter(const Texture& testTex, InspectorFlags inFlags);
void Demo_TextureAnnotations(const Texture& testTex, InspectorFlags inFlags);
} // namespace ImGuiTexInspect

/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/special/TextureInspector.h"

#include <imgui_tex_inspect_internal.h>
#include <tex_inspect_opengl.h>

#include "ImGuiTexInspect/DemoSnippets.h"

using namespace megamol::mmstd_gl::special;

TextureInspector::TextureInspector(const std::vector<std::string>& textures)
        : show_inspector_("Show", "Turn the texture inspector on or off.")
        , select_texture_("Texture", "Select which texture to be shown.")
        , tex_({nullptr, 0.f, 0.f})
        , flags_(0)
        , flip_x_(false)
        , flip_y_(true)
        , initiated_(false) {
    auto bp = new core::param::BoolParam(false);
    show_inspector_.SetParameter(bp);

    auto ep = new core::param::EnumParam(0);
    for (int i = 0; i < textures.size(); i++) {
        ep->SetTypePair(i, textures[i].c_str());
    }
    select_texture_.SetParameter(ep);
}

TextureInspector::TextureInspector()
        : show_inspector_("Show", "Turn the texture inspector on or off.")
        , select_texture_("Texture", "Select which texture to be shown.")
        , tex_({nullptr, 0.f, 0.f})
        , flags_(0)
        , flip_x_(false)
        , flip_y_(true)
        , initiated_(false) {
    auto bp = new core::param::BoolParam(false);
    show_inspector_.SetParameter(bp);

    auto ep = new core::param::EnumParam(0);
    ep->SetTypePair(0, "Default");
    select_texture_.SetParameter(ep);
}

TextureInspector::~TextureInspector() {}

//-------------------------------------------------------------------------
// [SECTION] MAIN SCENE WINDOW FUNCTION
//-------------------------------------------------------------------------

/**
 * void TextureInspector::ShowWindow
 */
void TextureInspector::ShowWindow() {
    if (!initiated_) {
        Init();
    }

    ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(1000, 1000), ImGuiCond_FirstUseEver);

    struct SceneConfig {
        const char* button_name; // Button text to display to user for a scene
        void (*draw_fn)(const ImGuiTexInspect::Texture& testTex,
            ImGuiTexInspect::InspectorFlags inFlags); // Function which implements the scene
    };

    const SceneConfig scenes[] = {
        {"Basics", &ImGuiTexInspect::Demo_ColorFilters},
        {"Color Matrix", &ImGuiTexInspect::Demo_ColorMatrix},
        {"Annotations", &ImGuiTexInspect::Demo_TextureAnnotations},
        {"Alpha Mode", &ImGuiTexInspect::Demo_AlphaMode},
        {"Wrap & Filter", &ImGuiTexInspect::Demo_WrapAndFilter},
    };

    if (ImGui::Begin("ImGuiTexInspect")) {
        ImGui::Text("Select Scene:");
        ImGui::Spacing();

        //Custom color values to example-select buttons to make them stand out
        ImGui::PushStyleColor(ImGuiCol_Button, (ImVec4)ImColor::HSV(0.59f, 0.7f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered, (ImVec4)ImColor::HSV(0.59f, 0.8f, 0.8f));
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, (ImVec4)ImColor::HSV(0.59f, 0.9f, 1.0f));

        // Draw row of buttons, one for each scene
        static int selected_scene = 0;
        for (int i = 0; i < IM_ARRAYSIZE(scenes); i++) {
            if (i != 0) {
                ImGui::SameLine();
            }
            if (ImGui::Button(scenes[i].button_name, ImVec2(140, 60))) {
                selected_scene = i;
            }
        }
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();
        ImGui::PopStyleColor();

        ImGui::Spacing();

        flags_ = 0; // reset flags
        if (flip_x_)
            SetFlag(flags_, ImGuiTexInspect::InspectorFlags_FlipX);
        if (flip_y_)
            SetFlag(flags_, ImGuiTexInspect::InspectorFlags_FlipY);

        // Call function to render currently example scene
        (*(scenes[selected_scene].draw_fn))({tex_.texture, ImVec2{tex_.x, tex_.y}}, flags_);

        ImGui::Separator();

        ImGui::Checkbox("Flip X", &flip_x_);
        ImGui::Checkbox("Flip Y", &flip_y_);
    }

    ImGui::End();
}

//-------------------------------------------------------------------------
// [SECTION] INIT & TEXTURE LOAD
//-------------------------------------------------------------------------

/**
 * void TextureInspector::Init
 */
void TextureInspector::Init() {
    ImGuiTexInspect::ImplOpenGL3_Init();
    ImGuiTexInspect::Init();
    ImGuiTexInspect::CreateContext();

    initiated_ = true;
}

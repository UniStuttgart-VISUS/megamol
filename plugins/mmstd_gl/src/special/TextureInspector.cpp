/**
 * The MIT License (MIT)
 *
 * Copyright (c) 2014-2021 Omar Cornut
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmstd_gl/special/TextureInspector.h"

#include "imgui_tex_inspect_internal.h"
#include "backends/tex_inspect_opengl.h"

using namespace megamol::mmstd_gl::special;

TextureInspector::TextureInspector(const std::vector<std::string>& textures)
        : show_inspector_("Show", "Turn the texture inspector on or off.")
        , select_texture_("Texture", "Select which texture to be shown.")
        , draw(nullptr)
        , tex_({nullptr, 0.f, 0.f})
        , flags_(0)
        , flip_x_(false)
        , flip_y_(true)
        , initiated_(false) {
    auto bp = new core::param::BoolParam(true);
    show_inspector_.SetParameter(bp);

    auto ep = new core::param::EnumParam(2);
    for (int i = 0; i < textures.size(); i++) {
        ep->SetTypePair(i, textures[i].c_str());
    }
    select_texture_.SetParameter(ep);
}

TextureInspector::~TextureInspector() {}


//-------------------------------------------------------------------------
// [SECTION] EXAMPLE USAGES
//-------------------------------------------------------------------------

/* Each of the following Scene* functions is a standalone demo showing an example
 * usage of ImGuiTexInspect.  You'll notice the structure is essentially the same in
 * all the examples, i.e. a call to BeginInspectorPanel & a call to
 * EndInspectorPanel, possibly with some annotation functions in between,
 * followed by some controls to manipulate the inspector.
 *
 * Each Scene* function corresponds to one of the large buttons at the top of
 * the scene window.
 */

/**
 * void TextureInspector::SceneColorFilters
 */
void TextureInspector::SceneColorFilters() {
    /* BeginInspectorPanel & EndInspectorPanel is all you need to draw an
     * inspector (assuming you are already in between an ImGui::Begin and
     * ImGui::End pair)
     * */

    if (ImGuiTexInspect::BeginInspectorPanel("##ColorFilters", this->tex_.texture, ImVec2(this->tex_.x, this->tex_.y), flags_)) {
        // Draw some text showing color value of each texel (you must be zoomed in to see this)
        ImGuiTexInspect::DrawAnnotations(ImGuiTexInspect::ValueText(ImGuiTexInspect::ValueText::BytesDec));
    }
    ImGuiTexInspect::EndInspectorPanel();

    // Now some ordinary ImGui elements to provide some explanation
    ImGui::BeginChild("Controls", ImVec2(600, 100));
    ImGui::TextWrapped("Basics:");
    ImGui::BulletText("Use mouse wheel to zoom in and out.  Click and drag to pan.");
    ImGui::BulletText("Use the scene select buttons at the top of the window to explore");
    ImGui::BulletText("Use the controls below to change basic color filtering options");
    ImGui::EndChild();


    /* DrawColorChannelSelector & DrawGridEditor are convenience functions that
     * draw ImGui controls to manipulate config of the most recently drawn
     * texture inspector
     **/
    ImGuiTexInspect::DrawColorChannelSelector();
    ImGui::SameLine(200);
    ImGuiTexInspect::DrawGridEditor();
}

//-------------------------------------------------------------------------

/**
 * void TextureInspector::SceneColorMatrix
 */
void TextureInspector::SceneColorMatrix() {
    if (ImGuiTexInspect::BeginInspectorPanel(
            "##ColorMatrix", this->tex_.texture, ImVec2(this->tex_.x, this->tex_.y), flags_)) {
        // Draw some text showing color value of each texel (you must be zoomed in to see this)
        ImGuiTexInspect::DrawAnnotations(ImGuiTexInspect::ValueText(ImGuiTexInspect::ValueText::BytesDec));
    }
    ImGuiTexInspect::EndInspectorPanel();

    ImGui::BeginGroup();
    ImGui::Text("Colour Matrix Editor:");
    // Draw Matrix editor to allow user to manipulate the ColorMatrix
    ImGuiTexInspect::DrawColorMatrixEditor();
    ImGui::EndGroup();

    ImGui::SameLine();

    // Provide some presets that can be used to set the ColorMatrix for example purposes
    ImGui::BeginGroup();
    ImGui::PushItemWidth(200);
    ImGui::Indent(50);
    const ImVec2 button_size = ImVec2(160, 0);
    ImGui::Text("Example Presets:");
    // clang-format off
    if (ImGui::Button("Negative", button_size))
    {
        // Matrix which inverts each of the red, green, blue channels and leaves Alpha untouched
        float matrix[] = {-1.000f,  0.000f,  0.000f,  0.000f, 
                           0.000f, -1.000f,  0.000f,  0.000f,
                           0.000f,  0.000f, -1.000f,  0.000f, 
                           0.000f,  0.000f,  0.000f,  1.000f};

        float color_offset[] = {1, 1, 1, 0};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(matrix, color_offset);
    }
    if (ImGui::Button("Swap Red & Blue", button_size))
    {
        // Matrix which swaps red and blue channels but leaves green and alpha untouched
        float matrix[] = { 0.000f,  0.000f,  1.000f,  0.000f, 
                           0.000f,  1.000f,  0.000f,  0.000f,
                           1.000f,  0.000f,  0.000f,  0.000f, 
                           0.000f,  0.000f,  0.000f,  1.000f};
        float color_offset[] = {0, 0, 0, 0};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(matrix, color_offset);
    }
    if (ImGui::Button("Alpha", button_size))
    {
        // Red, green and blue channels are set based on alpha value so that alpha = 1 shows as white. 
        // output alpha is set to 1
        float highlight_transparency_matrix[] = {0.000f, 0.000f, 0.000f, 0.000f,
                                               0.000f, 0.000f, 0.000f, 0.000f,
                                               0.000f, 0.000f, 0.000f, 0.000f, 
                                               1.000,  1.000,  1.000,  1.000f};
        float highlight_transparency_offset[] = {0, 0, 0, 1};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(highlight_transparency_matrix, highlight_transparency_offset);
    }
    if (ImGui::Button("Transparency", button_size))
    {
        // Red, green and blue channels are scaled by 0.1f. Low alpha values are shown as magenta
        float highlight_transparency_matrix[] = {0.100f,  0.100f,  0.100f,  0.000f, 
                                               0.100f,  0.100f,  0.100f,  0.000f,
                                               0.100f,  0.100f,  0.100f,  0.000f, 
                                              -1.000f,  0.000f, -1.000f,  0.000f};
        float highlight_transparency_offset[] = {1, 0, 1, 1};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(highlight_transparency_matrix, highlight_transparency_offset);
    }
    if (ImGui::Button("Default", button_size))
    {
        // Default "identity" matrix that doesn't modify colors at all
        float matrix[] = {1.000f, 0.000f, 0.000f, 0.000f, 
                          0.000f, 1.000f, 0.000f, 0.000f,
                          0.000f, 0.000f, 1.000f, 0.000f, 
                          0.000f, 0.000f, 0.000f, 1.000f};

        float color_offset[] = {0, 0, 0, 0};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(matrix, color_offset);
    }
    // clang-format on
    ImGui::PopItemWidth();
    ImGui::EndGroup();
}

/**
 * void TextureInspector::SceneAlphaMode
 */
void TextureInspector::SceneAlphaMode() {
    if (ImGuiTexInspect::BeginInspectorPanel(
            "##AlphaMode", this->tex_.texture, ImVec2(this->tex_.x, this->tex_.y), flags_)) {
        // Add annotations here
    }
    ImGuiTexInspect::EndInspectorPanel();
    ImGuiTexInspect::DrawAlphaModeSelector();
}

/**
 * void TextureInspector::SceneWrapAndFilter
 */
void TextureInspector::SceneWrapAndFilter() {
    static bool show_wrap = false;
    static bool force_nearest_texel = true;

    if (ImGuiTexInspect::BeginInspectorPanel(
            "##WrapAndFilter", this->tex_.texture, ImVec2(this->tex_.x, this->tex_.y), flags_)) {}

    ImGuiTexInspect::InspectorFlags flags = 0;

    if (show_wrap)
        flags |= ImGuiTexInspect::InspectorFlags_ShowWrap;
    if (!force_nearest_texel)
        flags |= ImGuiTexInspect::InspectorFlags_NoForceFilterNearest;

    ImGuiTexInspect::CurrentInspector_SetFlags(flags, ~flags);
    ImGuiTexInspect::EndInspectorPanel();

    ImGui::BeginChild("Controls", ImVec2(600, 0));
    ImGui::TextWrapped(
        "The following option can be enabled to render texture outside of the [0,1] UV range, what you actually "
        "see outside of this range will depend on the mode of the texture. For example you may see the texture repeat, "
        "or "
        "it might be clamped to the colour of the edge pixels.\nIn this scene the texture is set to wrap.");
    ImGui::Checkbox("Show Wrapping Mode", &show_wrap);

    ImGui::TextWrapped("The following option is enabled by default and forces a nearest texel filter, implemented at "
                       "the shader level. "
                       "By disabling this you can the currently set mode for this texture.");
    ImGui::Checkbox("Force Nearest Texel", &force_nearest_texel);
    ImGui::EndChild();
}

// This class is used in SceneTextureAnnotations to show the process of creating a new texture annotation.
class CustomAnnotationExample {
public:
    void DrawAnnotation(ImDrawList* drawList, ImVec2 texel, ImGuiTexInspect::Transform2D texelsToPixels, ImVec4 value) {
        /* A silly example to show the process of creating a new annotation
         * We'll see which primary colour is the dominant colour in the texel
         * then draw a different shape for each primary colour.  The radius
         * will be based on the overall brightness.
         */
        int num_segments;

        if (value.x > value.y && value.x > value.z) {
            // Red pixel - draw a triangle!
            num_segments = 3;
        } else {
            if (value.y > value.z) {
                // Green pixel - draw a diamond!
                num_segments = 4;
            } else {
                // Blue pixel - draw a hexagon!
                num_segments = 6;
            }
        }

        // Don't go larger than whole texel
        const float max_radius = texelsToPixels.Scale.x * 0.5f;

        // Scale radius based on texel brightness
        const float radius = max_radius * (value.x + value.y + value.z) / 3;
        drawList->AddNgon(texelsToPixels * texel, radius, 0xFFFFFFFF, num_segments);
    }
};

/**
* void TextureInspector::SceneTextureAnnotations
*/
void TextureInspector::SceneTextureAnnotations() {
    static bool annotation_enabled_arrow = true;
    static bool annotation_enabled_value_text = false;
    static bool annotation_enabled_custom_example = false;

    static ImGuiTexInspect::ValueText::Format text_format = ImGuiTexInspect::ValueText::BytesHex;

    const int max_annotated_texels = 1000;

    if (ImGuiTexInspect::BeginInspectorPanel(
            "##TextureAnnotations", this->tex_.texture, ImVec2(this->tex_.x, this->tex_.y), flags_)) {
        // Draw the currently enabled annotations...
        if (annotation_enabled_arrow) {
            ImGuiTexInspect::DrawAnnotations(
                ImGuiTexInspect::Arrow().UsePreset(ImGuiTexInspect::Arrow::NormalMap), max_annotated_texels);
        }

        if (annotation_enabled_value_text) {
            ImGuiTexInspect::DrawAnnotations(ImGuiTexInspect::ValueText(text_format), max_annotated_texels);
        }

        if (annotation_enabled_custom_example) {
            ImGuiTexInspect::DrawAnnotations(CustomAnnotationExample(), max_annotated_texels);
        }
    }
    ImGuiTexInspect::EndInspectorPanel();

    // Checkboxes to toggle each type of annotation on and off
    ImGui::BeginChild("Controls", ImVec2(600, 0));
    ImGui::Checkbox("Arrow (Hint: zoom in on the normal-map part of the texture)", &annotation_enabled_arrow);
    ImGui::Checkbox("Value Text", &annotation_enabled_value_text);
    ImGui::Checkbox("Custom Annotation Example", &annotation_enabled_custom_example);
    ImGui::EndChild();

    if (annotation_enabled_value_text) {
        // Show a combo to select the text formatting mode
        ImGui::SameLine();
        ImGui::BeginGroup();
        const char* text_options[] = {"Hex String", "Bytes in Hex", "Bytes in Decimal", "Floats"};
        ImGui::SetNextItemWidth(200);
        int text_format_int = (int)(text_format);
        ImGui::Combo("Text Mode", &text_format_int, text_options, IM_ARRAYSIZE(text_options));
        text_format = (ImGuiTexInspect::ValueText::Format)text_format_int;
        ImGui::EndGroup();
    }
}

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
        void (TextureInspector::*draw_fn)(); // Function which implements the scene
    };

    const SceneConfig scenes[] = {
        {"Basics", &TextureInspector::SceneColorFilters},
        {"Color Matrix", &TextureInspector::SceneColorMatrix},
        {"Annotations", &TextureInspector::SceneTextureAnnotations},
        {"Alpha Mode", &TextureInspector::SceneAlphaMode},
        {"Wrap & Filter", &TextureInspector::SceneWrapAndFilter},
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
        this->draw = scenes[selected_scene].draw_fn;
        (this->*draw)();

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

#include "DemoSnippets.h"

#include "imgui_tex_inspect_internal.h"

using namespace ImGuiTexInspect;

// Source: https://github.com/andyborrell/imgui_tex_inspect/blob/main/imgui_tex_inspect_demo.cpp#L56-L97
void ImGuiTexInspect::Demo_ColorFilters(const Texture& testTex, InspectorFlags inFlags)
{
    /* BeginInspectorPanel & EndInspectorPanel is all you need to draw an
     * inspector (assuming you are already in between an ImGui::Begin and 
     * ImGui::End pair) 
     * */

    if (ImGuiTexInspect::BeginInspectorPanel("##ColorFilters", testTex.texture, testTex.size, inFlags))
    {
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

// Source: https://github.com/andyborrell/imgui_tex_inspect/blob/main/imgui_tex_inspect_demo.cpp#L108-L188
void ImGuiTexInspect::Demo_ColorMatrix(const Texture& testTex, InspectorFlags inFlags)
{
    if (ImGuiTexInspect::BeginInspectorPanel("##ColorMatrix", testTex.texture, testTex.size, inFlags))
    {
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
    const ImVec2 buttonSize = ImVec2(160, 0);
    ImGui::Text("Example Presets:");
    // clang-format off
    if (ImGui::Button("Negative", buttonSize))
    {
        // Matrix which inverts each of the red, green, blue channels and leaves Alpha untouched
        float matrix[] = {-1.000f,  0.000f,  0.000f,  0.000f, 
                           0.000f, -1.000f,  0.000f,  0.000f,
                           0.000f,  0.000f, -1.000f,  0.000f, 
                           0.000f,  0.000f,  0.000f,  1.000f};

        float colorOffset[] = {1, 1, 1, 0};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(matrix, colorOffset);
    }
    if (ImGui::Button("Swap Red & Blue", buttonSize))
    {
        // Matrix which swaps red and blue channels but leaves green and alpha untouched
        float matrix[] = { 0.000f,  0.000f,  1.000f,  0.000f, 
                           0.000f,  1.000f,  0.000f,  0.000f,
                           1.000f,  0.000f,  0.000f,  0.000f, 
                           0.000f,  0.000f,  0.000f,  1.000f};
        float colorOffset[] = {0, 0, 0, 0};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(matrix, colorOffset);
    }
    if (ImGui::Button("Alpha", buttonSize))
    {
        // Red, green and blue channels are set based on alpha value so that alpha = 1 shows as white. 
        // output alpha is set to 1
        float highlightTransparencyMatrix[] = {0.000f, 0.000f, 0.000f, 0.000f,
                                               0.000f, 0.000f, 0.000f, 0.000f,
                                               0.000f, 0.000f, 0.000f, 0.000f, 
                                               1.000,  1.000,  1.000,  1.000f};
        float highlightTransparencyOffset[] = {0, 0, 0, 1};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(highlightTransparencyMatrix, highlightTransparencyOffset);
    }
    if (ImGui::Button("Transparency", buttonSize))
    {
        // Red, green and blue channels are scaled by 0.1f. Low alpha values are shown as magenta
        float highlightTransparencyMatrix[] = {0.100f,  0.100f,  0.100f,  0.000f, 
                                               0.100f,  0.100f,  0.100f,  0.000f,
                                               0.100f,  0.100f,  0.100f,  0.000f, 
                                              -1.000f,  0.000f, -1.000f,  0.000f};
        float highlightTransparencyOffset[] = {1, 0, 1, 1};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(highlightTransparencyMatrix, highlightTransparencyOffset);
    }
    if (ImGui::Button("Default", buttonSize))
    {
        // Default "identity" matrix that doesn't modify colors at all
        float matrix[] = {1.000f, 0.000f, 0.000f, 0.000f, 
                          0.000f, 1.000f, 0.000f, 0.000f,
                          0.000f, 0.000f, 1.000f, 0.000f, 
                          0.000f, 0.000f, 0.000f, 1.000f};

        float colorOffset[] = {0, 0, 0, 0};
        ImGuiTexInspect::CurrentInspector_SetColorMatrix(matrix, colorOffset);
    }
    // clang-format on
    ImGui::PopItemWidth();
    ImGui::EndGroup();
}

// Source: https://github.com/andyborrell/imgui_tex_inspect/blob/main/imgui_tex_inspect_demo.cpp#L194-L202
void ImGuiTexInspect::Demo_AlphaMode(const Texture& testTex, InspectorFlags inFlags)
{
    if (ImGuiTexInspect::BeginInspectorPanel("##AlphaModeDemo", testTex.texture, testTex.size, inFlags))
    {
        // Add annotations here
    }
    ImGuiTexInspect::EndInspectorPanel();
    ImGuiTexInspect::DrawAlphaModeSelector();
}

// Source: https://github.com/andyborrell/imgui_tex_inspect/blob/80ffc679e8f3f477d861d7a806e072098e94158c/imgui_tex_inspect_demo.cpp#L208-L236
void ImGuiTexInspect::Demo_WrapAndFilter(const Texture& testTex, InspectorFlags inFlags)
{
    static bool showWrap = false;
    static bool forceNearestTexel = true;

    if (ImGuiTexInspect::BeginInspectorPanel("##WrapAndFilter", testTex.texture, testTex.size), inFlags)
    {
    }
    ImGuiTexInspect::InspectorFlags flags = 0;

    if (showWrap)
        flags |= ImGuiTexInspect::InspectorFlags_ShowWrap;
    if (!forceNearestTexel)
        flags |= ImGuiTexInspect::InspectorFlags_NoForceFilterNearest;

    ImGuiTexInspect::CurrentInspector_SetFlags(flags, ~flags);
    ImGuiTexInspect::EndInspectorPanel();

    ImGui::BeginChild("Controls", ImVec2(600, 0));
    ImGui::TextWrapped("The following option can be enabled to render texture outside of the [0,1] UV range, what you actually "
                       "see outside of this range will depend on the mode of the texture. For example you may see the texture repeat, or "
                       "it might be clamped to the colour of the edge pixels.\nIn this demo the texture is set to wrap.");
    ImGui::Checkbox("Show Wrapping Mode", &showWrap);

    ImGui::TextWrapped("The following option is enabled by default and forces a nearest texel filter, implemented at the shader level. "
                       "By disabling this you can the currently set mode for this texture.");
    ImGui::Checkbox("Force Nearest Texel", &forceNearestTexel);
    ImGui::EndChild();
}

// Source: https://github.com/andyborrell/imgui_tex_inspect/blob/main/imgui_tex_inspect_demo.cpp#L238-L328
// This class is used in Demo_TextureAnnotations to show the process of creating a new texture annotation.
class CustomAnnotationExample
{
    public:
        void DrawAnnotation(ImDrawList *drawList, ImVec2 texel, ImGuiTexInspect::Transform2D texelsToPixels, ImVec4 value)
        {
            /* A silly example to show the process of creating a new annotation 
             * We'll see which primary colour is the dominant colour in the texel 
             * then draw a different shape for each primary colour.  The radius 
             * will be based on the overall brightness. 
             */
            int numSegments;

            if (value.x > value.y && value.x > value.z)
            {
                // Red pixel - draw a triangle!
                numSegments = 3;
            }
            else
            {
                if (value.y > value.z)
                {
                    // Green pixel - draw a diamond!
                    numSegments = 4;
                }
                else
                {
                    // Blue pixel - draw a hexagon!
                    numSegments = 6;
                }
            }

            // Don't go larger than whole texel
            const float maxRadius = texelsToPixels.Scale.x * 0.5f;

            // Scale radius based on texel brightness
            const float radius = maxRadius * (value.x + value.y + value.z) / 3;
            drawList->AddNgon(texelsToPixels * texel, radius, 0xFFFFFFFF, numSegments);
        }
};

void ImGuiTexInspect::Demo_TextureAnnotations(const Texture& testTex, InspectorFlags inFlags)
{
    static bool annotationEnabled_arrow = true;
    static bool annotationEnabled_valueText = false;
    static bool annotationEnabled_customExample = false;

    static ImGuiTexInspect::ValueText::Format textFormat = ImGuiTexInspect::ValueText::BytesHex;

    const int maxAnnotatedTexels = 1000;

    if (ImGuiTexInspect::BeginInspectorPanel("##TextureAnnotations", testTex.texture, testTex.size, inFlags))
    {
        // Draw the currently enabled annotations...
        if (annotationEnabled_arrow)
        {
            ImGuiTexInspect::DrawAnnotations(ImGuiTexInspect::Arrow().UsePreset(ImGuiTexInspect::Arrow::NormalMap), maxAnnotatedTexels);
        }

        if (annotationEnabled_valueText)
        {
            ImGuiTexInspect::DrawAnnotations(ImGuiTexInspect::ValueText(textFormat), maxAnnotatedTexels);
        }

        if (annotationEnabled_customExample)
        {
            ImGuiTexInspect::DrawAnnotations(CustomAnnotationExample(), maxAnnotatedTexels);
        }
    }
    ImGuiTexInspect::EndInspectorPanel();

    // Checkboxes to toggle each type of annotation on and off
    ImGui::BeginChild("Controls", ImVec2(600, 0));
    ImGui::Checkbox("Arrow (Hint: zoom in on the normal-map part of the texture)", &annotationEnabled_arrow);
    ImGui::Checkbox("Value Text",                                                  &annotationEnabled_valueText);
    ImGui::Checkbox("Custom Annotation Example",                                   &annotationEnabled_customExample);
    ImGui::EndChild();

    if (annotationEnabled_valueText)
    {
        // Show a combo to select the text formatting mode
        ImGui::SameLine();
        ImGui::BeginGroup();
        const char *textOptions[] = {"Hex String", "Bytes in Hex", "Bytes in Decimal", "Floats"};
        ImGui::SetNextItemWidth(200);
        int textFormatInt = (int)(textFormat);
        ImGui::Combo("Text Mode", &textFormatInt, textOptions, IM_ARRAYSIZE(textOptions));
        textFormat = (ImGuiTexInspect::ValueText::Format)textFormatInt;
        ImGui::EndGroup();
    }
}

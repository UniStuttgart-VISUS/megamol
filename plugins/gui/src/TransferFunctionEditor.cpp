/*
 * TransferFunctionEditor.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TransferFunctionEditor.h"
#include "ColorPalettes.h"

#include <array>
#include <cmath>

using namespace megamol::gui;
using namespace megamol::core;


/**
 * Cubehelix colormap, ported from Fortran77
 *
 * @param t Function domain between [0;1]
 * @param start Start color (1=red, 2=green, 3=blue; e.g. 0.5=purple)
 * @param rots Rotations in color (typically -1.5 to 1.5, e.g. -1.0 is
 *             one blue->green->red cycle);
 * @param hue Hue intensity scaling (in the range 0.0 (B+W) to 1.0 to be
 *            strictly correct, larger values may be OK with particular
 *            start/end colors);
 * @param gamma Set the gamma correction for intensity.
 */
std::array<double, 3> CubeHelixRGB(double t, double start, double rots, double hue, double gamma) {
    const double PI = 3.141592653589793238463; // Fuck C++!

    double angle = 2.0 * PI * (start / 3.0 + 1 + rots * t);
    double fract = std::pow(t, gamma);
    double amp = hue * fract * (1 - fract) / 2.0;

    double r = fract + amp * (-0.14861 * std::cos(angle) + 1.78277 * std::sin(angle));
    double g = fract + amp * (-0.29227 * std::cos(angle) - 0.90649 * std::sin(angle));
    double b = fract + amp * (+1.97294 * std::cos(angle));

    r = std::max(std::min(r, 1.0), 0.0);
    g = std::max(std::min(g, 1.0), 0.0);
    b = std::max(std::min(b, 1.0), 0.0);

    return {r, g, b};
}

/**
 * Transform from Hue to RGB.
 */
std::array<double, 3> HueToRGB(double hue) {
    std::array<double, 3> color;
    color[0] = hue;
    color[1] = hue + 1.0 / 3.0;
    color[2] = hue + 2.0 / 3.0;
    for (size_t i = 0; i < color.size(); ++i) {
        color[i] = std::max(0.0, std::min(6.0 * std::abs(color[i] - std::floor(color[i]) - 0.5) - 1.0, 1.0));
    }
    return std::move(color);
}

using PresetGenerator = std::function<void(param::TransferFunctionParam::TFNodeType&, size_t)>;

PresetGenerator CubeHelixAdapter(double start, double rots, double hue, double gamma) {
    return [=](auto& nodes, auto n) {
        nodes.clear();
        for (size_t i = 0; i < n; ++i) {
            auto t = i / static_cast<double>(n - 1);
            auto color = CubeHelixRGB(t, start, rots, hue, gamma);
            nodes.push_back({
                static_cast<float>(color[0]),
                static_cast<float>(color[1]),
                static_cast<float>(color[2]),
                1.0f,
                static_cast<float>(t),
                0.05f,
            });
        }
    };
}

template <size_t PaletteSize> PresetGenerator ColormapAdapter(const float palette[PaletteSize][3]) {
    const double LastIndex = static_cast<double>(PaletteSize - 1);
    return [=](auto& nodes, auto n) {
        nodes.clear();
        for (size_t i = 0; i < n; ++i) {
            auto t = i / static_cast<double>(n - 1);

            // Linear interpolation from palette.
            size_t i0 = std::floor(t * LastIndex);
            size_t i1 = std::ceil(t * LastIndex);
            double it = std::fmod(t * LastIndex, LastIndex);
            double r[2] = {static_cast<double>(palette[i0][0]), static_cast<double>(palette[i1][0])};
            double g[2] = {static_cast<double>(palette[i0][1]), static_cast<double>(palette[i1][1])};
            double b[2] = {static_cast<double>(palette[i0][2]), static_cast<double>(palette[i1][2])};

            nodes.push_back({
                static_cast<float>(std::max(0.0, std::min((1.0 - it) * r[0] + it * r[1], 1.0))),
                static_cast<float>(std::max(0.0, std::min((1.0 - it) * g[0] + it * g[1], 1.0))),
                static_cast<float>(std::max(0.0, std::min((1.0 - it) * b[0] + it * b[1], 1.0))),
                1.0f,
                static_cast<float>(t),
                0.05f,
            });
        }
    };
}

void RampAdapter(param::TransferFunctionParam::TFNodeType& nodes, size_t n) {
    nodes.clear();
    std::array<float, TFP_VAL_CNT> zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.05f};
    std::array<float, TFP_VAL_CNT> one = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.05f};
    nodes.emplace_back(zero);
    nodes.emplace_back(one);
}

void RainbowAdapter(param::TransferFunctionParam::TFNodeType& nodes, size_t n) {
    nodes.clear();
    for (size_t i = 0; i < n; ++i) {
        auto t = i / static_cast<double>(n - 1);
        auto color = HueToRGB(t);
        nodes.push_back({
            static_cast<float>(color[0]),
            static_cast<float>(color[1]),
            static_cast<float>(color[2]),
            1.0f,
            static_cast<float>(t),
            0.05f,
        });
    }
}

std::array<std::tuple<std::string, PresetGenerator>, 12> PRESETS = {
    std::make_tuple("Select...", [](auto& nodes, auto n) {}), std::make_tuple("Ramp", RampAdapter),
    std::make_tuple("Hue rotation (rainbow, harmful)", RainbowAdapter),
    std::make_tuple("Inferno", ColormapAdapter<256>(InfernoColorMap)),
    std::make_tuple("Magma", ColormapAdapter<256>(MagmaColorMap)),
    std::make_tuple("Plasma", ColormapAdapter<256>(PlasmaColorMap)),
    std::make_tuple("Viridis", ColormapAdapter<256>(ViridisColorMap)),
    std::make_tuple("Parula", ColormapAdapter<256>(ParulaColorMap)),
    std::make_tuple("Cubehelix (default)", CubeHelixAdapter(0.5, -1.5, 1.0, 1.0)),
    std::make_tuple("Cubehelix (default, colorful)", CubeHelixAdapter(0.5, -1.5, 1.5, 1.0)),
    std::make_tuple("Cubehelix (default, de-pinked)", CubeHelixAdapter(0.5, -1.0, 1.0, 1.0)),
    std::make_tuple("Cool-Warm (diverging)", ColormapAdapter<257>(CoolWarmColorMap))};


TransferFunctionEditor::TransferFunctionEditor(void)
    : utils()
    , activeParameter(nullptr)
    , range({0.0f, 1.0f})
    , mode(param::TransferFunctionParam::InterpolationMode::LINEAR)
    , textureSize(256)
    , textureId(0)
    , textureInvalid(false)
    , activeChannels{false, false, false, false}
    , currentNode(0)
    , currentChannel(0)
    , currentDragChange()
    , immediateMode(false)
    , showOptions(true) {
    // Init transfer function colors
    this->nodes.clear();
    std::array<float, TFP_VAL_CNT> zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.05f};
    std::array<float, TFP_VAL_CNT> one = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.05f};
    this->nodes.emplace_back(zero);
    this->nodes.emplace_back(one);

    this->widget_buffer.min_range = this->range[0];
    this->widget_buffer.max_range = this->range[1];
    this->widget_buffer.tex_size = textureSize;
    this->widget_buffer.gauss_sigma = zero[5];
    this->widget_buffer.range_value = zero[4];

    this->textureInvalid = true;
}

void TransferFunctionEditor::SetTransferFunction(const std::string& tfs) {
    if (activeParameter == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TransferFunctionEditor] Missing active parameter to edit");
        return;
    }

    bool ok = megamol::core::param::TransferFunctionParam::ParseTransferFunction(
        tfs, this->nodes, this->mode, this->textureSize, this->range);
    if (!ok) {
        vislib::sys::Log::DefaultLog.WriteWarn("[TransferFunctionEditor] Could parse transfer function");
        return;
    }

    // Check for required initial node data
    assert(this->nodes.size() > 1);

    this->currentNode = 0;
    this->currentChannel = 0;
    this->currentDragChange = ImVec2(0.0f, 0.0f);

    this->widget_buffer.min_range = this->range[0];
    this->widget_buffer.max_range = this->range[1];
    this->widget_buffer.tex_size = this->textureSize;

    this->widget_buffer.range_value =
        (this->nodes[this->currentNode][4] * (this->range[1] - this->range[0])) + this->range[0];
    this->widget_buffer.gauss_sigma = this->nodes[this->currentNode][5];

    this->textureInvalid = true;
}

bool TransferFunctionEditor::GetTransferFunction(std::string& tfs) {
    return param::TransferFunctionParam::DumpTransferFunction(
        tfs, this->nodes, this->mode, this->textureSize, this->range);
}

bool TransferFunctionEditor::DrawTransferFunctionEditor(void) {
    assert(ImGui::GetCurrentContext() != nullptr);
    assert(this->nodes.size() > 1);

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    if (this->activeParameter == nullptr) {
        const char* message = "Changes have no effect.\n"
                              "Please set a transfer function parameter.\n";
        ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), message);
    }

    // Test if selected node is still in range
    if (this->nodes.size() <= this->currentNode) {
        this->currentNode = 0;
    }

    const float tfw_height = 28.0f;
    const float tfw_item_width = ImGui::GetContentRegionAvailWidth() * 0.75f;
    const float canvas_height = 150.0f;
    const float canvas_width = tfw_item_width;
    ImGui::PushItemWidth(tfw_item_width); // set general proportional item width

    this->drawTextureBox(ImVec2(tfw_item_width, 30.0f));

    ImGui::SameLine();
    if (ImGui::ArrowButton("Options", this->showOptions ? ImGuiDir_Down : ImGuiDir_Up)) {
        this->showOptions = !this->showOptions;
    }
    if (!this->showOptions) {
        return false;
    }

    ImGui::Separator();

    // Interval range -----------------------------------------------------
    ImGui::PushItemWidth(tfw_item_width * 0.5 - style.ItemInnerSpacing.x);

    ImGui::InputFloat("###min", &this->widget_buffer.min_range, 1.0f, 10.0f, "%.6f", ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        this->range[0] =
            (this->widget_buffer.min_range < this->range[1]) ? (this->widget_buffer.min_range) : (this->range[0]);
        if (this->range[0] >= this->range[1]) {
            this->range[0] = this->range[1] - 0.000001f;
        }
        this->widget_buffer.min_range = this->range[0];
        this->textureInvalid = true;
    }
    ImGui::SameLine();

    ImGui::InputFloat("###max", &this->widget_buffer.max_range, 1.0f, 10.0f, "%.6f", ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        this->range[1] =
            (this->widget_buffer.max_range > this->range[0]) ? (this->widget_buffer.max_range) : (this->range[1]);
        if (this->range[0] >= this->range[1]) {
            this->range[1] = this->range[0] + 0.000001f;
        }
        this->widget_buffer.max_range = this->range[1];
        this->textureInvalid = true;
    }
    ImGui::SameLine();
    ImGui::PopItemWidth();
    ImGui::SetCursorPosX(tfw_item_width + style.ItemSpacing.x + style.ItemInnerSpacing.x);
    ImGui::Text("Value Range");

    // Value slider -------------------------------------------------------
    this->widget_buffer.range_value =
        (this->nodes[this->currentNode][4] * (this->range[1] - this->range[0])) + this->range[0];
    if (ImGui::SliderFloat("Selected Value", &this->widget_buffer.range_value, this->range[0], this->range[1])) {
        float new_x = (this->widget_buffer.range_value - this->range[0]) / (this->range[1] - this->range[0]);
        if (this->currentNode == 0) {
            new_x = 0.0f;
        } else if (this->currentNode == (this->nodes.size() - 1)) {
            new_x = 1.0f;
        } else if (new_x < this->nodes[this->currentNode - 1][4]) {
            new_x = this->nodes[this->currentNode - 1][4];
        } else if (new_x > this->nodes[this->currentNode + 1][4]) {
            new_x = this->nodes[this->currentNode + 1][4];
        }
        this->nodes[this->currentNode][4] = new_x;
        this->textureInvalid = true;
    }
    std::string help = "[Ctrl-Click] for keyboard input";
    this->utils.HelpMarkerToolTip(help);

    // Sigma slider -------------------------------------------------------
    if (this->mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
        if (ImGui::SliderFloat("Selected Sigma", &this->widget_buffer.gauss_sigma, 0.0f, 1.0f)) {
            this->nodes[this->currentNode][5] = this->widget_buffer.gauss_sigma;
            this->textureInvalid = true;
        }
        std::string help = "[Ctrl-Click] for keyboard input";
        this->utils.HelpMarkerToolTip(help);
    }

    // Plot ---------------------------------------------------------------

    this->drawFunctionPlot(ImVec2(canvas_width, canvas_height));

    // Color channels -----------------------------------------------------
    ImGui::Checkbox("Red", &this->activeChannels[0]);
    ImGui::SameLine(tfw_item_width * 0.26);
    ImGui::Checkbox("Green", &this->activeChannels[1]);
    ImGui::SameLine(tfw_item_width * 0.49);
    ImGui::Checkbox("Blue", &this->activeChannels[2]);
    ImGui::SameLine(tfw_item_width * 0.725);
    ImGui::Checkbox("Alpha", &this->activeChannels[3]);
    ImGui::SameLine(tfw_item_width + style.ItemSpacing.x + style.ItemInnerSpacing.x);
    ImGui::Text("Color Channels");

    // Color editor for selected node -------------------------------------
    float edit_col[4] = {this->nodes[this->currentNode][0], this->nodes[this->currentNode][1],
        this->nodes[this->currentNode][2], this->nodes[this->currentNode][3]};
    ImGuiColorEditFlags numberColorFlags =
        ImGuiColorEditFlags_RGB | ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_Float;
    if (ImGui::ColorEdit4("Selected Color", edit_col, numberColorFlags)) {
        this->nodes[this->currentNode][0] = edit_col[0];
        this->nodes[this->currentNode][1] = edit_col[1];
        this->nodes[this->currentNode][2] = edit_col[2];
        this->nodes[this->currentNode][3] = edit_col[3];
        this->textureInvalid = true;
    }
    help = "[Click] on the colored square to open a color picker.\n"
           "[CTRL+Click] on individual component to input value.\n"
           "[Right-Click] on the individual color widget to show options.";
    this->utils.HelpMarkerToolTip(help);


    // Interpolation mode -------------------------------------------------
    std::map<param::TransferFunctionParam::InterpolationMode, std::string> opts;
    opts[param::TransferFunctionParam::InterpolationMode::LINEAR] = "Linear";
    opts[param::TransferFunctionParam::InterpolationMode::GAUSS] = "Gauss";
    int opts_cnt = opts.size();
    if (ImGui::BeginCombo("Interpolation", opts[this->mode].c_str())) {
        for (int i = 0; i < opts_cnt; ++i) {
            if (ImGui::Selectable(opts[(param::TransferFunctionParam::InterpolationMode)i].c_str(),
                    (this->mode == (param::TransferFunctionParam::InterpolationMode)i))) {
                this->mode = (param::TransferFunctionParam::InterpolationMode)i;
                this->textureInvalid = true;
            }
        }
        ImGui::EndCombo();
    }

    // Presets -------------------------------------------------
    if (ImGui::BeginCombo("Load Preset", std::get<0>(PRESETS[0]).c_str())) {
        for (auto preset : PRESETS) {
            if (ImGui::Selectable(std::get<0>(preset).c_str())) {
                std::get<1>(preset)(this->nodes, this->textureSize);
                this->textureInvalid = true;
            }
        }
        ImGui::EndCombo();
    }

    // Texture size -------------------------------------------------------
    ImGui::InputInt("Texture Size", &this->widget_buffer.tex_size, 1, 10, ImGuiInputTextFlags_None);
    if (ImGui::IsItemDeactivatedAfterEdit()) {
        this->textureSize = (UINT)std::max(1, this->widget_buffer.tex_size);
        this->widget_buffer.tex_size = this->textureSize;
        this->textureInvalid = true;
    }

    // --------------------------------------------------------------------

    // Create current texture data
    bool imm_apply_tex_modified = this->textureInvalid;
    if (this->textureInvalid) {
        if (this->mode == param::TransferFunctionParam::InterpolationMode::LINEAR) {
            param::TransferFunctionParam::LinearInterpolation(this->texturePixels, this->textureSize, this->nodes);
        } else if (this->mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
            param::TransferFunctionParam::GaussInterpolation(this->texturePixels, this->textureSize, this->nodes);
        }

        // Delete old texture.
        if (this->textureId != 0) {
            glDeleteTextures(1, &this->textureId);
        }
        this->textureId = 0;

        // Upload texture.
        glGenTextures(1, &this->textureId);
        glBindTexture(GL_TEXTURE_2D, this->textureId);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, textureSize, 1, 0, GL_RGBA, GL_FLOAT, this->texturePixels.data());

        glBindTexture(GL_TEXTURE_2D, 0);

        this->textureInvalid = false;
    }

    bool shouldApply = false;

    // Return true for current changes being applied
    if (ImGui::Button("Apply")) {
        shouldApply = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Auto-apply", &this->immediateMode)) {
        shouldApply = this->immediateMode;
    }

    if (this->immediateMode && imm_apply_tex_modified) {
        shouldApply = true;
    }

    if (shouldApply) {
        if (this->activeParameter != nullptr) {
            std::string tf;
            if (this->GetTransferFunction(tf)) {
                this->activeParameter->SetValue(tf);
            }
        }
    }

    return shouldApply;
}


void TransferFunctionEditor::drawTextureBox(const ImVec2& size) {
    ImVec2 pos = ImGui::GetCursorScreenPos();
    const size_t textureSize = this->texturePixels.size() / 4;

    if (textureSize == 0 || this->textureId == 0) {
        // Reserve layout space and draw a black background rectangle.
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImGui::Dummy(size);
        drawList->AddRectFilled(pos, ImVec2(pos.x + size.x, pos.y + size.y), IM_COL32(0, 0, 0, 255), 0.0f, 10);
    } else {
        // Draw texture as image.
        ImGui::Image(reinterpret_cast<ImTextureID>(this->textureId), size);
    }

    // Draw tooltip, if requested.
    if (ImGui::IsItemHovered()) {
        float xPx = ImGui::GetMousePos().x - pos.x - ImGui::GetScrollX();
        float xU = xPx / size.x;
        float xValue = xU * (this->range[1] - this->range[0]) + this->range[0];
        ImGui::BeginTooltip();
        ImGui::Text("%f Absolute Value\n%f Normalized Value", xValue, xU);
        ImGui::EndTooltip();
    }
}


void TransferFunctionEditor::drawFunctionPlot(const ImVec2& size) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* drawList = ImGui::GetWindowDrawList();

    ImVec2 canvas_pos = ImGui::GetCursorScreenPos(); // ImDrawList API uses screen coordinates!
    ImVec2 canvas_size = size;
    if (canvas_size.x < 50.0f) canvas_size.x = 100.0f;
    if (canvas_size.y < 50.0f) canvas_size.y = 50.0f;
    ImVec2 mouse_cur_pos = io.MousePos; // current mouse position

    ImVec4 tmp_frameBkgrd = style.Colors[ImGuiCol_FrameBg];
    tmp_frameBkgrd.w = 1.0f;

    ImU32 alpha_line_col = IM_COL32(255, 255, 255, 255);
    // Adapt color for alpha line depending on lightness of background
    float L = (std::max(tmp_frameBkgrd.x, std::max(tmp_frameBkgrd.y, tmp_frameBkgrd.z)) +
                  std::min(tmp_frameBkgrd.x, std::min(tmp_frameBkgrd.y, tmp_frameBkgrd.z))) /
              2.0f;
    if (L > 0.5f) {
        alpha_line_col = IM_COL32(0, 0, 0, 255);
    }
    ImU32 frameBkgrd = ImGui::ColorConvertFloat4ToU32(tmp_frameBkgrd);

    const float line_width = 2.0f;
    const float point_radius = 6.0f;
    const float point_border_width = 1.5f;
    const int circle_subdiv = 12;
    ImVec2 delta_border = style.ItemInnerSpacing;

    // Draw a background rectangle.
    drawList->AddRectFilled(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), frameBkgrd);

    // Clip lines within the canvas (if we resize it, etc.)
    drawList->PushClipRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), true);

    const std::array<ImU32, 4> channelColors{{
        IM_COL32(255, 0, 0, 255),
        IM_COL32(0, 255, 0, 255),
        IM_COL32(0, 0, 255, 255),
        alpha_line_col,
    }};

    int selected_node = -1;
    int selected_chan = -1;
    ImVec2 selected_delta = ImVec2(0.0f, 0.0f);
    // For each enabled color channel
    for (size_t c = 0; c < channelColors.size(); ++c) {
        if (!this->activeChannels[c]) continue;

        const float pointAndBorderRadius = point_radius + point_border_width - 2.0f;

        // Draw lines.
        drawList->PathClear();
        for (size_t i = 0; i < this->nodes.size(); ++i) {
            ImVec2 point = ImVec2(canvas_pos.x + this->nodes[i][4] * canvas_size.x,
                canvas_pos.y + (1.0f - this->nodes[i][c]) * canvas_size.y);
            if (this->mode == param::TransferFunctionParam::InterpolationMode::LINEAR) {
                drawList->PathLineTo(point);
            } else if (this->mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
                const float ga = this->nodes[i][c];
                const float gb = this->nodes[i][4];
                const float gc = this->nodes[i][5];
                const int step = 3; // pixel step width in x direction
                float x, g;
                ImVec2 pos;

                for (int p = 0; p < (int)canvas_size.x + step; p += step) {
                    x = (float)(p) / canvas_size.x;
                    g = param::TransferFunctionParam::gauss(x, ga, gb, gc);
                    pos =
                        ImVec2(canvas_pos.x + (x * canvas_size.x), canvas_pos.y + canvas_size.y - (g * canvas_size.y));
                    drawList->PathLineTo(pos);
                }
                drawList->PathStroke(channelColors[c], false, line_width);
            }

            // Test for intersection of mouse position with node.
            ImVec2 d = ImVec2(point.x - mouse_cur_pos.x, point.y - mouse_cur_pos.y);
            if (sqrtf((d.x * d.x) + (d.y * d.y)) <= pointAndBorderRadius) {
                selected_node = i;
                selected_chan = c;
                selected_delta = d;
            }
        }
        drawList->PathStroke(channelColors[c], false, line_width);

        // Draw node circles.
        for (size_t i = 0; i < this->nodes.size(); ++i) {
            if ((i != selected_node || c != selected_chan) && i != this->currentNode) {
                continue;
            }

            ImVec2 point = ImVec2(canvas_pos.x + this->nodes[i][4] * canvas_size.x,
                canvas_pos.y + (1.0f - this->nodes[i][c]) * canvas_size.y);
            ImU32 pointColor =
                ImGui::ColorConvertFloat4ToU32(ImVec4(this->nodes[i][0], this->nodes[i][1], this->nodes[i][2], 1.0));
            ImU32 pointBorderColor = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_TextDisabled]);
            if (i == this->currentNode) {
                pointBorderColor = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
            }

            // Draw node point
            drawList->AddCircle(point, pointAndBorderRadius, pointBorderColor, circle_subdiv, point_border_width);
            drawList->AddCircleFilled(point, point_radius, pointColor, 12);
        }
    }
    drawList->PopClipRect();

    // Process mouse interaction
    ImGui::InvisibleButton("plot", canvas_size); // Needs to catch mouse input
    if (ImGui::IsItemHovered() && (mouse_cur_pos.x > (canvas_pos.x - delta_border.x)) &&
        (mouse_cur_pos.y > (canvas_pos.y - delta_border.y)) &&
        (mouse_cur_pos.x < (canvas_pos.x + canvas_size.x + delta_border.x)) &&
        (mouse_cur_pos.y < (canvas_pos.y + canvas_size.y + delta_border.y))) {

        if (io.MouseClicked[0]) {
            // Left Click -> Change selected node selected node
            if (selected_node >= 0) {
                this->currentNode = selected_node;
                this->currentChannel = selected_chan;
                this->currentDragChange = selected_delta;

                this->widget_buffer.range_value =
                    (this->nodes[this->currentNode][4] * (this->range[1] - this->range[0])) + this->range[0];
                this->widget_buffer.gauss_sigma = this->nodes[this->currentNode][5];
            }
        } else if (io.MouseDown[0]) {
            // Left Move -> Move selected node
            float new_x = (mouse_cur_pos.x - canvas_pos.x + this->currentDragChange.x) / canvas_size.x;
            new_x = std::max(0.0f, std::min(new_x, 1.0f));
            if (this->currentNode == 0) {
                new_x = 0.0f;
            } else if (this->currentNode == (this->nodes.size() - 1)) {
                new_x = 1.0f;
            } else if (new_x < this->nodes[this->currentNode - 1][4]) {
                new_x = this->nodes[this->currentNode - 1][4];
            } else if (new_x > this->nodes[this->currentNode + 1][4]) {
                new_x = this->nodes[this->currentNode + 1][4];
            }
            this->nodes[this->currentNode][4] = new_x;
            this->widget_buffer.range_value =
                (this->nodes[this->currentNode][4] * (this->range[1] - this->range[0])) + this->range[0];

            float new_y = 1.0f - ((mouse_cur_pos.y - canvas_pos.y + this->currentDragChange.y) / canvas_size.y);
            new_y = std::max(0.0f, std::min(new_y, 1.0f));

            if (this->activeChannels[0] && (this->currentChannel == 0)) {
                this->nodes[this->currentNode][0] = new_y;
            }
            if (this->activeChannels[1] && (this->currentChannel == 1)) {
                this->nodes[this->currentNode][1] = new_y;
            }
            if (this->activeChannels[2] && (this->currentChannel == 2)) {
                this->nodes[this->currentNode][2] = new_y;
            }
            if (this->activeChannels[3] && (this->currentChannel == 3)) {
                this->nodes[this->currentNode][3] = new_y;
            }
            this->textureInvalid = true;

        } else if (io.MouseClicked[1]) {
            // Right Click -> Add/delete Node
            if (selected_node < 0) {
                // Add new at current position
                float new_x = (mouse_cur_pos.x - canvas_pos.x) / canvas_size.x;
                new_x = std::max(0.0f, std::min(new_x, 1.0f));

                float new_y = 1.0f - ((mouse_cur_pos.y - canvas_pos.y) / canvas_size.y);
                new_y = std::max(0.0f, std::min(new_y, 1.0f));

                for (auto it = this->nodes.begin(); it != this->nodes.end(); ++it) {
                    if (new_x < (*it)[4]) {
                        // New nodes can only be inserted between two exisintng ones,
                        // so there is always a node before and after
                        std::array<float, TFP_VAL_CNT> prev_col = (*(it - 1));
                        std::array<float, TFP_VAL_CNT> fol_col = (*it);
                        std::array<float, TFP_VAL_CNT> new_col = {(prev_col[0] + fol_col[0]) / 2.0f,
                            (prev_col[1] + fol_col[1]) / 2.0f, (prev_col[2] + fol_col[2]) / 2.0f,
                            (prev_col[3] + fol_col[3]) / 2.0f, new_x, 0.05f};

                        if (this->activeChannels[0]) {
                            new_col[0] = new_y;
                        }
                        if (this->activeChannels[1]) {
                            new_col[1] = new_y;
                        }
                        if (this->activeChannels[2]) {
                            new_col[2] = new_y;
                        }
                        if (this->activeChannels[3]) {
                            new_col[3] = new_y;
                        }
                        this->nodes.insert(it, new_col);
                        this->textureInvalid = true;
                        break;
                    }
                }
            } else {
                // Delete currently hovered
                if ((selected_node > 0) &&
                    (selected_node < (this->nodes.size() - 1))) { // First and last node can't be deleted
                    this->nodes.erase(this->nodes.begin() + selected_node);
                    if (this->currentNode >= selected_node) {
                        this->currentNode = (unsigned int)std::max(0, (int)this->currentNode - 1);
                    }
                    this->textureInvalid = true;
                }
            }
        }
    }
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    ImGui::Text("Function Plot");
    this->utils.HelpMarkerToolTip(
        "First and last node are always present\nwith fixed value 0 and 1.\n[Left-Click] Select "
        "Node\n[Left-Drag] Move Node\n[Right-Click] Add/Delete Node");
}
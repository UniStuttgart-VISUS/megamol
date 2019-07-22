/*
 * TransferFunctionEditor.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TransferFunctionEditor.h"


using namespace megamol::gui;
using namespace megamol::core;


TransferFunctionEditor::TransferFunctionEditor(void)
    : Popup()
    , activeParameter(nullptr)
    , data()
    , range({0.0f, 1.0f})
    , mode(param::TransferFunctionParam::InterpolationMode::LINEAR)
    , textureSize(128)
    , texturePixels()
    , textureInvalid(false)
    , activeChannels{false, false, false, true}
    , currentNode(0)
    , currentChannel(0)
    , currentDragChange()
    , immediateMode(false) {

    // Init transfer function colors
    this->data.clear();
    std::array<float, TFP_VAL_CNT> zero = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.05f};
    std::array<float, TFP_VAL_CNT> one = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.05f};
    this->data.emplace_back(zero);
    this->data.emplace_back(one);
    this->textureInvalid = true;
}

bool TransferFunctionEditor::SetTransferFunction(const std::string& tfs) {
    if (activeParameter == nullptr) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[TransferFunctionEditor] Load parameter before editing transfer function");
        return false;
    }

    bool result = megamol::core::param::TransferFunctionParam::ParseTransferFunction(
        tfs, this->data, this->mode, this->textureSize, this->range);
    if (result) {
        this->textureInvalid = true;
    }

    return result;
}

bool TransferFunctionEditor::GetTransferFunction(std::string& tfs) {
    return megamol::core::param::TransferFunctionParam::DumpTransferFunction(
        tfs, this->data, this->mode, this->textureSize, this->range);
}

bool TransferFunctionEditor::DrawTransferFunctionEditor(void) {
    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    if (this->activeParameter == nullptr) {
        ImGui::TextColored(ImVec4(0.9f, 0.0f, 0.0f, 1.0f), "Changes have no effect.\n"
                                                           "Please load transfer function parameter.\n");
        ImGui::Separator();
    }

    const float tfw_height = 28.0f;
    const float tfw_item_width = ImGui::GetContentRegionAvailWidth() * 0.75f;
    const float canvas_height = 150.0f;
    const float canvas_width = tfw_item_width;
    ImGui::PushItemWidth(tfw_item_width); // set general proportional item width

    // Check for required initial node data
    assert(this->data.size() > 1);

    // Check if selected node is still in range
    if (this->data.size() <= this->currentNode) {
        this->currentNode = 0;
    }

    // Check range (delta should not equal zero)
    if (this->range[0] == this->range[1]) {
        this->range[1] += 0.000001f;
    }

    ImDrawList* draw_list = ImGui::GetWindowDrawList();

    // ------------------------------------------------------------------------

    // Draw current transfer function texture
    const float texture_height = 30.0f;
    ImVec2 texture_pos = ImGui::GetCursorScreenPos();
    ImVec2 rect_size = ImVec2(tfw_item_width / (float)this->textureSize, texture_height);
    ImGui::InvisibleButton("texture", ImVec2(tfw_item_width, rect_size.y));
    if ((this->textureSize * 4) == this->texturePixels.size()) { // Wait for updated texture data
        for (unsigned int i = 0; i < this->textureSize; ++i) {
            ImU32 rect_col = ImGui::ColorConvertFloat4ToU32(ImVec4(this->texturePixels[4 * i],
                this->texturePixels[4 * i + 1], this->texturePixels[4 * i + 2], this->texturePixels[4 * i + 3]));
            ImVec2 rect_pos_a = ImVec2(texture_pos.x + (float)i * rect_size.x, texture_pos.y);
            ImVec2 rect_pos_b = ImVec2(rect_pos_a.x + rect_size.x, rect_pos_a.y + rect_size.y);
            draw_list->AddRectFilled(rect_pos_a, rect_pos_b, rect_col, 0.0f, 10);
        }
    }
    // ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    // ImGui::Text("1D Texture");

    // ------------------------------------------------------------------------

    // Select color channels
    ImGui::Checkbox("Red", &this->activeChannels[0]);
    ImGui::SameLine();
    ImGui::Checkbox("Green", &this->activeChannels[1]);
    ImGui::SameLine();
    ImGui::Checkbox("Blue", &this->activeChannels[2]);
    ImGui::SameLine();
    ImGui::Checkbox("Alpha", &this->activeChannels[3]);
    ImGui::SameLine(tfw_item_width + style.ItemSpacing.x + style.ItemInnerSpacing.x);
    ImGui::Text("Color Channels");

    // Plot
    ImVec2 canvas_pos = ImGui::GetCursorScreenPos(); // ImDrawList API uses screen coordinates!
    ImVec2 canvas_size = ImVec2(canvas_width, canvas_height);
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

    const float point_radius = 10.0f;
    const float point_border = 4.0f;
    const int circle_subdiv = 12;
    ImVec2 delta_border = style.ItemInnerSpacing;

    // Draw rectangle for graph
    draw_list->AddRectFilledMultiColor(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y),
        frameBkgrd, frameBkgrd, frameBkgrd, frameBkgrd);

    // Clip lines within the canvas (if we resize it, etc.)
    draw_list->PushClipRect(canvas_pos, ImVec2(canvas_pos.x + canvas_size.x, canvas_pos.y + canvas_size.y), true);

    int selected_node = -1;
    int selected_chan = -1;
    ImVec2 selected_delta = ImVec2(0.0f, 0.0f);
    for (int i = 0; i < this->data.size(); ++i) {
        ImU32 point_col = ImGui::ColorConvertFloat4ToU32(
            ImVec4(this->data[i][0], this->data[i][1], this->data[i][2], this->data[i][3]));

        // For each enabled color channel
        for (int c = 0; c < 4; ++c) {
            if (!this->activeChannels[c]) continue;

            // Define line color
            ImU32 line_col = alpha_line_col; // for c == 3 (alpha)
            if (c == 0) line_col = IM_COL32(255, 0, 0, 255);
            if (c == 1) line_col = IM_COL32(0, 255, 0, 255);
            if (c == 2) line_col = IM_COL32(0, 0, 255, 255);

            // Draw lines/curves ...
            ImVec2 point_cur_pos = ImVec2(canvas_pos.x + this->data[i][4] * canvas_size.x,
                canvas_pos.y + (1.0f - this->data[i][c]) * canvas_size.y);

            if (this->mode == param::TransferFunctionParam::InterpolationMode::LINEAR) {
                if (i < (this->data.size() - 1)) {
                    ImVec2 point_next_pos = ImVec2(canvas_pos.x + this->data[i + 1][4] * canvas_size.x,
                        canvas_pos.y + (1.0f - this->data[i + 1][c]) * canvas_size.y);

                    draw_list->AddLine(point_cur_pos, point_next_pos, line_col, 4.0f);
                }
            } else if (this->mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
                const float ga = this->data[i][c];
                const float gb = this->data[i][4];
                const float gc = this->data[i][5];
                const int step = 3; // step width in x direction
                float x0, x1;
                float g0, g1;
                float last_g1 = 0.0f;

                for (int p = 0; p < (int)canvas_size.x; p += step) {
                    x0 = (float)p / canvas_size.x;
                    x1 = (float)(p + step) / canvas_size.x;

                    g0 = last_g1;
                    if (p == 0) {
                        x0 = (float)(-step) / canvas_size.x;
                        g0 = param::TransferFunctionParam::gauss(x0, ga, gb, gc);
                    }
                    ImVec2 pos0 = ImVec2(
                        canvas_pos.x + (x0 * canvas_size.x), canvas_pos.y + canvas_size.y - (g0 * canvas_size.y));

                    if (p == ((int)canvas_size.x - 1)) {
                        x1 = (float)(canvas_size.x + step) / canvas_size.x;
                    }
                    g1 = param::TransferFunctionParam::gauss(x1, ga, gb, gc);
                    ImVec2 pos1 = ImVec2(
                        canvas_pos.x + (x1 * canvas_size.x), canvas_pos.y + canvas_size.y - (g1 * canvas_size.y));
                    last_g1 = g1;

                    draw_list->AddLine(pos0, pos1, line_col, 4.0f);
                }
            }

            // Draw node point
            ImU32 point_border_col = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_TextDisabled]);
            if (i == this->currentNode) {
                point_border_col = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
            }
            draw_list->AddCircleFilled(point_cur_pos, point_radius, frameBkgrd, circle_subdiv);
            float point_radius_full = point_radius + point_border - 2.0f;
            draw_list->AddCircle(point_cur_pos, point_radius_full, point_border_col, circle_subdiv, point_border);
            draw_list->AddCircleFilled(point_cur_pos, point_radius, point_col, 12);

            // Check intersection of mouse with node point
            ImVec2 d = ImVec2(point_cur_pos.x - mouse_cur_pos.x, point_cur_pos.y - mouse_cur_pos.y);
            if (sqrtf((d.x * d.x) + (d.y * d.y)) <= point_radius_full) {
                selected_node = i;
                selected_chan = c;
                selected_delta = d;
            }
        }
    }
    draw_list->PopClipRect();

    // Process mouse interaction
    ImGui::InvisibleButton("plot", canvas_size);
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
            }
        } else if (io.MouseDown[0]) {
            // Left Move -> Move selected node
            float new_x = (mouse_cur_pos.x - canvas_pos.x + this->currentDragChange.x) / canvas_size.x;
            new_x = std::max(0.0f, std::min(new_x, 1.0f));
            if (this->currentNode == 0) {
                new_x = 0.0f;
            } else if (this->currentNode == (this->data.size() - 1)) {
                new_x = 1.0f;
            } else if ((new_x <= this->data[this->currentNode - 1][4]) ||
                       (new_x >= this->data[this->currentNode + 1][4])) {
                new_x = this->data[this->currentNode][4];
            }
            this->data[this->currentNode][4] = new_x;

            float new_y = 1.0f - ((mouse_cur_pos.y - canvas_pos.y + this->currentDragChange.y) / canvas_size.y);
            new_y = std::max(0.0f, std::min(new_y, 1.0f));

            if (this->activeChannels[0] && (this->currentChannel == 0)) {
                this->data[this->currentNode][0] = new_y;
            }
            if (this->activeChannels[1] && (this->currentChannel == 1)) {
                this->data[this->currentNode][1] = new_y;
            }
            if (this->activeChannels[2] && (this->currentChannel == 2)) {
                this->data[this->currentNode][2] = new_y;
            }
            if (this->activeChannels[3] && (this->currentChannel == 3)) {
                this->data[this->currentNode][3] = new_y;
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

                for (auto it = this->data.begin(); it != this->data.end(); ++it) {
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
                        this->data.insert(it, new_col);
                        this->textureInvalid = true;
                        break;
                    }
                }
            } else {
                // Delete currently hovered
                if ((selected_node > 0) &&
                    (selected_node < (this->data.size() - 1))) { // First and last node can't be deleted
                    this->data.erase(this->data.begin() + selected_node);
                    if (this->currentNode >= selected_node) {
                        this->currentNode = (unsigned int)std::max(0, (int)this->currentNode - 1);
                    }
                    this->textureInvalid = true;
                }
            }
        }
    }
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    ImGui::Text("Transfer Function");
    this->HelpMarkerToolTip("[Left-Click] Select Node\n[Left-Drag] Move Node\n[Right-Click] Add/Delete Node");


    // Scale value to given range
    float delta_range = (this->range[1] - this->range[0]);
    float value = (this->data[this->currentNode][4] * delta_range) - this->range[0];

    // Value slider
    if (ImGui::SliderFloat("Selected Value", &value, this->range[0], this->range[1])) {
        float new_x = value;
        if (this->currentNode == 0) {
            new_x = 0.0f;
        } else if (this->currentNode == (this->data.size() - 1)) {
            new_x = 1.0f;
        } else if ((new_x <= this->data[this->currentNode - 1][4]) || (new_x >= this->data[this->currentNode + 1][4])) {
            new_x = this->data[this->currentNode][4];
        }
        this->data[this->currentNode][4] = new_x;
        this->textureInvalid = true;
    }
    std::string help = "[Ctrl-Click] for keyboard input";
    this->HelpMarkerToolTip(help);

    // Sigma slider
    if (this->mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
        float sigma = this->data[this->currentNode][5];
        if (ImGui::SliderFloat("Selected Sigma", &sigma, 0.0f, 1.0f)) {
            this->data[this->currentNode][5] = sigma;
            this->textureInvalid = true;
        }
        std::string help = "[Ctrl-Click] for keyboard input";
        this->HelpMarkerToolTip(help);
    }

    // Color editor for selected node
    float edit_col[4] = {this->data[this->currentNode][0], this->data[this->currentNode][1],
        this->data[this->currentNode][2], this->data[this->currentNode][3]};
    if (ImGui::ColorEdit4("Selected Color", edit_col)) {
        this->data[this->currentNode][0] = edit_col[0];
        this->data[this->currentNode][1] = edit_col[1];
        this->data[this->currentNode][2] = edit_col[2];
        this->data[this->currentNode][3] = edit_col[3];
        this->textureInvalid = true;
    }
    help = "[Click] on the colored square to open a color picker.\n"
           "[CTRL+Click] on individual component to input value.\n"
           "[Right-Click] on the individual color widget to show options.";
    this->HelpMarkerToolTip(help);

    // Get new texture size
    int tfw_texsize = (int)this->textureSize;
    if (ImGui::InputInt("Texture Size", &tfw_texsize, 1, 10, ImGuiInputTextFlags_EnterReturnsTrue)) {
        this->textureSize = (UINT)std::max(1, tfw_texsize);
        this->textureInvalid = true;
    }

    // Select interpolation mode (Linear, Gauss, ...)
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

    // Create current texture data
    bool imm_apply_tex_modified = this->textureInvalid;
    if (this->textureInvalid) {
        if (this->mode == param::TransferFunctionParam::InterpolationMode::LINEAR) {
            param::TransferFunctionParam::LinearInterpolation(this->texturePixels, this->textureSize, this->data);
        } else if (this->mode == param::TransferFunctionParam::InterpolationMode::GAUSS) {
            param::TransferFunctionParam::GaussInterpolation(this->texturePixels, this->textureSize, this->data);
        }
        this->textureInvalid = false;
    }

    // Return true for current changes being applied
    bool ret_val = false;
    if (ImGui::Button("Apply Changes")) {
        ret_val = true;
    }
    ImGui::SameLine();
    if (ImGui::Checkbox("Apply Changes Immediately", &this->immediateMode)) {
        ret_val = this->immediateMode;
    }

    if (this->immediateMode && imm_apply_tex_modified) {
        ret_val = true;
    }

    if (ret_val) {
        if (this->activeParameter != nullptr) {
            std::string tf;
            if (this->GetTransferFunction(tf)) {
                this->activeParameter->SetValue(tf);
            }
        }
    }

    return ret_val;
}

/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "TransferFunctionEditor.h"

#include "graph/Parameter.h"
#include "gui_utils.h"
#include "widgets/ButtonWidgets.h"
#include "widgets/ColorPalettes.h"


#define TF_FLOAT_EPS 1e-5f


using namespace megamol;
using namespace megamol::gui;
using namespace megamol::core;
using namespace megamol::core::param;


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
    const double PI = 3.141592653589793238463;

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
    std::array<double, 3> color = {hue, (hue + 1.0 / 3.0), (hue + 2.0 / 3.0)};
    for (size_t i = 0; i < color.size(); ++i) {
        color[i] = std::max(0.0, std::min(6.0 * std::abs(color[i] - std::floor(color[i]) - 0.5) - 1.0, 1.0));
    }
    return color;
}

using PresetGenerator = std::function<void(TransferFunctionParam::NodeVector_t&, size_t)>;

PresetGenerator CubeHelixAdapter(double start, double rots, double hue, double gamma) {
    return [=](auto& nodes, auto n) {
        nodes.clear();
        for (size_t i = 0; i < n; ++i) {
            auto t = static_cast<double>(i) / static_cast<double>(n - 1);
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

template<size_t PaletteSize, bool NearestNeighbor = false>
PresetGenerator ColormapAdapter(const float palette[PaletteSize][3]) {
    auto LastIndex = static_cast<double>(PaletteSize - 1);
    return [=](auto& nodes, auto n) {
        nodes.clear();
        for (size_t i = 0; i < n; ++i) {
            auto t = static_cast<double>(i) / static_cast<double>(n - 1);

            // Linear interpolation from palette.
            auto i0 = static_cast<size_t>(std::floor(t * LastIndex));
            auto i1 = static_cast<size_t>(std::ceil(t * LastIndex));
            double unused;
            double it = std::modf(t * LastIndex, &unused);
            if (NearestNeighbor) {
                it = std::round(it);
            }

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

void RampAdapter(TransferFunctionParam::NodeVector_t& nodes, size_t n) {
    nodes.clear();
    nodes.push_back({0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.05f});
    nodes.push_back({1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.05f});
}

void RainbowAdapter(TransferFunctionParam::NodeVector_t& nodes, size_t n) {
    nodes.clear();
    for (size_t i = 0; i < n; ++i) {
        auto t = static_cast<double>(i) / static_cast<double>(n - 1);
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

std::array<std::tuple<std::string, PresetGenerator>, 21> PRESETS = {
    std::make_tuple("Select...", [](auto& nodes, auto n) {}),
    std::make_tuple("Ramp", RampAdapter),
    std::make_tuple("Cividis", ColormapAdapter<256>(CividisColorMap)),
    std::make_tuple("Hue rotation (rainbow, harmful)", RainbowAdapter),
    std::make_tuple("Inferno", ColormapAdapter<256>(InfernoColorMap)),
    std::make_tuple("Magma", ColormapAdapter<256>(MagmaColorMap)),
    std::make_tuple("Plasma", ColormapAdapter<256>(PlasmaColorMap)),
    std::make_tuple("Viridis", ColormapAdapter<256>(ViridisColorMap)),
    std::make_tuple("Parula", ColormapAdapter<256>(ParulaColorMap)),
    std::make_tuple("Cubehelix (default)", CubeHelixAdapter(0.5, -1.5, 1.0, 1.0)),
    std::make_tuple("Cubehelix (default, colorful)", CubeHelixAdapter(0.5, -1.5, 1.5, 1.0)),
    std::make_tuple("Cubehelix (default, de-pinked)", CubeHelixAdapter(0.5, -1.0, 1.0, 1.0)),
    std::make_tuple("Cool-Warm (diverging)", ColormapAdapter<257>(CoolWarmColorMap)),
    std::make_tuple("8-class Accent", ColormapAdapter<8, true>(AccentMap)),
    std::make_tuple("8-class Dark2", ColormapAdapter<8, true>(Dark2Map)),
    std::make_tuple("12-class Paired", ColormapAdapter<12, true>(PairedMap)),
    std::make_tuple("9-class Pastel1", ColormapAdapter<9, true>(Pastel1Map)),
    std::make_tuple("8-class Pastel2", ColormapAdapter<8, true>(Pastel2Map)),
    std::make_tuple("9-class Set1", ColormapAdapter<9, true>(Set1Map)),
    std::make_tuple("8-class Set2", ColormapAdapter<8, true>(Set2Map)),
    std::make_tuple("12-class Set3", ColormapAdapter<12, true>(Set3Map)),
};

// ----------------------------------------------------------------------------

TransferFunctionEditor::TransferFunctionEditor(const std::string& window_name, bool windowed)
        : AbstractWindow(window_name, AbstractWindow::WINDOW_ID_TRANSFER_FUNCTION)
        , windowed_mode(windowed)
        , connected_parameter_ptr(nullptr)
        , nodes()
        , range({0.0f, 1.0f})
        , last_range({0.0f, 1.0f})
        , range_overwrite(false)
        , interpolation_mode(TransferFunctionParam::InterpolationMode::LINEAR)
        , reload_texture(true)
        , texture_size(256)
        , pending_changes(true)
        , immediate_mode(false)
        , active_color_channels{true, true, true, true}
        , selected_channel_index(GUI_INVALID_ID)
        , selected_node_index(GUI_INVALID_ID)
        , selected_node_drag_delta()
        , show_options(true)
        , widget_buffer()
        , flip_legend(false)
        , check_once_force_set_overwrite_range(true)
        , plot_paint_mode(false)
        , plot_dragging(false)
        , request_parameter_name_connect()
        , win_view_minimized(false)
        , win_view_vertical(false)
        , win_connected_param_name()
        , win_tfe_reset(false)
        , tooltip()
        , image_widget_linear()
        , image_widget_nearest() {

    this->widget_buffer.left_range = this->range[0];
    this->widget_buffer.right_range = this->range[1];
    this->widget_buffer.tex_size = this->texture_size;
    this->widget_buffer.gauss_sigma = 0.05f;
    this->widget_buffer.range_value = 0.0f;

    // Load ramp as initial preset
    RampAdapter(this->nodes, this->texture_size);

    // Configure TRANSFER FUNCTION Window
    this->win_config.flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs;
    this->win_config.hotkey =
        megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F8, core::view::Modifier::NONE);
}


void TransferFunctionEditor::SetTransferFunction(
    const std::string& tfs, bool connected_parameter_mode, bool full_init) {

    if (connected_parameter_mode && (this->connected_parameter_ptr == nullptr)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Missing active parameter to edit");
        return;
    }

    if (full_init) {
        this->check_once_force_set_overwrite_range = true;
    }
    this->selected_node_index = GUI_INVALID_ID;
    this->selected_channel_index = GUI_INVALID_ID;
    this->selected_node_drag_delta = ImVec2(0.0f, 0.0f);

    unsigned int new_tex_size = 0;
    std::array<float, 2> new_range = {0.0f, 1.0f};
    TransferFunctionParam::NodeVector_t new_nodes;
    TransferFunctionParam::InterpolationMode new_interpolation_mode;
    if (!TransferFunctionParam::GetParsedTransferFunctionData(
            tfs, new_nodes, new_interpolation_mode, new_tex_size, new_range)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Could not parse transfer function");
        return;
    }
    bool tf_changed = false;
    if (this->interpolation_mode != new_interpolation_mode) {
        this->interpolation_mode = new_interpolation_mode;
        tf_changed = true;
    }
    if (this->nodes != new_nodes) {
        this->nodes = new_nodes;
        tf_changed = true;
    }
    if (this->range != new_range) {
        this->range_overwrite = false;
        this->last_range = this->range;
        this->range = new_range;
        tf_changed = true;
    }

    if (this->check_once_force_set_overwrite_range) {
        this->range_overwrite = !TransferFunctionParam::IgnoreProjectRange(tfs);
        this->last_range = this->range;
        this->check_once_force_set_overwrite_range = false;
        tf_changed = true;
    }

    if (this->texture_size != static_cast<int>(new_tex_size)) {
        this->texture_size = static_cast<int>(new_tex_size);
        this->widget_buffer.tex_size = this->texture_size;
        tf_changed = true;
    }

    if (tf_changed) {
        this->widget_buffer.left_range = this->range[0];
        this->widget_buffer.right_range = this->range[1];
        if ((this->selected_node_index != GUI_INVALID_ID) && (this->selected_node_index < this->nodes.size())) {
            this->widget_buffer.range_value =
                (this->nodes[this->selected_node_index][4] * (this->range[1] - this->range[0])) + this->range[0];
            this->widget_buffer.gauss_sigma = this->nodes[this->selected_node_index][5];
        } else {
            this->widget_buffer.range_value = this->widget_buffer.left_range;
            this->widget_buffer.gauss_sigma = 0.05f;
        }
        this->reload_texture = true;
    }
}

bool TransferFunctionEditor::GetTransferFunction(std::string& tfs) {
    return TransferFunctionParam::GetDumpedTransferFunction(tfs, this->nodes, this->interpolation_mode,
        static_cast<unsigned int>(this->texture_size), this->range, !this->range_overwrite);
}


void TransferFunctionEditor::SetConnectedParameter(Parameter* param_ptr, const std::string& param_full_name) {
    this->connected_parameter_ptr = nullptr;
    this->win_connected_param_name = "";
    if (param_ptr != nullptr) {
        if (param_ptr->Type() == ParamType_t::TRANSFERFUNCTION) {
            this->connected_parameter_ptr = param_ptr;
            this->win_connected_param_name = param_full_name;
            this->SetTransferFunction(std::get<std::string>(this->connected_parameter_ptr->GetValue()), true, true);
        } else {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Wrong parameter type. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        }
    }
}


bool TransferFunctionEditor::Update() {

    if (this->win_tfe_reset) {
        this->SetMinimized(this->win_view_minimized);
        this->SetVertical(this->win_view_vertical);
        this->request_parameter_name_connect = this->win_connected_param_name;
        this->win_tfe_reset = false;
    }

    // Change window flags depending on current view of transfer function editor
    if (this->IsMinimized()) {
        this->win_config.flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoResize |
                                 ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoTitleBar |
                                 ImGuiWindowFlags_NoNavInputs;
    } else {
        this->win_config.flags = ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs;
    }
    this->win_view_minimized = this->IsMinimized();
    this->win_view_vertical = this->IsVertical();

    return true;
}


bool TransferFunctionEditor::TransferFunctionEditor::Draw() {

    std::string help;

    ImGui::BeginGroup();
    ImGui::PushID("TransferFunctionEditor");

    if (this->windowed_mode && (!this->IsAnyParameterConnected())) {
        const char* message = "Changes have no effect.\n"
                              "No transfer function parameter connected for edit.\n";
        ImGui::TextColored(GUI_COLOR_TEXT_ERROR, message);
    }

    assert(ImGui::GetCurrentContext() != nullptr);

    ImGuiStyle& style = ImGui::GetStyle();

    const float height = (30.0f * megamol::gui::gui_scaling.Get());
    const float width = (300.0f * megamol::gui::gui_scaling.Get());
    const float tfw_item_width = ImGui::GetContentRegionAvail().x * 0.775f;
    ImGui::PushItemWidth(tfw_item_width); // set general proportional item width
    ImVec2 image_size = ImVec2(width, height);
    if (this->show_options) {
        image_size = ImVec2(tfw_item_width, height);
        if (this->flip_legend) {
            image_size = ImVec2(height, width);
        }
    } else {
        if (this->flip_legend) {
            image_size = ImVec2(height, width);
        }
    }

    ImGui::BeginGroup();
    this->drawTextureBox(image_size);
    this->drawScale(ImGui::GetCursorScreenPos(), image_size);
    ImGui::EndGroup();

    ImGui::SameLine();

    if (ImGui::ArrowButton("Options_", this->show_options ? ImGuiDir_Down : ImGuiDir_Up)) {
        this->show_options = !this->show_options;
    }

    if (this->show_options) {
        ImGui::Separator();

        // Legend alignment ---------------------------------------------------
        ImGui::BeginGroup();
        if (ImGui::RadioButton("Vertical", this->flip_legend)) {
            this->flip_legend = true;
            this->reload_texture = true;
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("Horizontal", !this->flip_legend)) {
            this->flip_legend = false;
            this->reload_texture = true;
        }
        ImGui::SameLine(tfw_item_width + style.ItemInnerSpacing.x + ImGui::GetScrollX());
        ImGui::TextUnformatted("Legend Alignment");
        ImGui::EndGroup();

        // Interval range -----------------------------------------------------
        ImGui::PushItemWidth(tfw_item_width * 0.5f - style.ItemSpacing.x);
        gui_utils::PushReadOnly(!this->range_overwrite);
        ImGui::InputFloat("###min", &this->widget_buffer.left_range, 1.0f, 10.0f, "%.6f", ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            this->range[0] = this->widget_buffer.left_range;
            this->reload_texture = true;
        }
        ImGui::SameLine();
        ImGui::InputFloat("###max", &this->widget_buffer.right_range, 1.0f, 10.0f, "%.6f", ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            this->range[1] = this->widget_buffer.right_range;
            this->reload_texture = true;
        }
        gui_utils::PopReadOnly(!this->range_overwrite);
        ImGui::PopItemWidth();
        ImGui::SameLine(0.0f, (style.ItemSpacing.x + style.ItemInnerSpacing.x));
        ImGui::TextUnformatted("Value Range");

        if (megamol::gui::ButtonWidgets::ToggleButton("Overwrite Value Range", this->range_overwrite)) {
            if (this->range_overwrite) {
                // Save last range before overwrite
                this->last_range = this->range;
            } else {
                // Reset range to last range before overwrite was enabled
                this->range = this->last_range;
                this->widget_buffer.left_range = this->range[0];
                this->widget_buffer.right_range = this->range[1];
                if ((this->selected_node_index != GUI_INVALID_ID) && (this->selected_node_index < this->nodes.size())) {
                    this->widget_buffer.range_value =
                        (this->nodes[this->selected_node_index][4] * (this->range[1] - this->range[0])) +
                        this->range[0];
                } else {
                    this->widget_buffer.range_value = this->widget_buffer.left_range;
                }
            }
            this->reload_texture = true;
        }
        help = "[Enable] for overwriting value range propagated from connected module(s).\n"
               "[Disable] for recovery of last value range propagated from connected module(s).";
        this->tooltip.Marker(help);

        // START selected NODE options ----------------------------------------
        bool node_selected =
            ((this->selected_node_index != GUI_INVALID_ID) && (this->selected_node_index < this->nodes.size()));
        megamol::gui::gui_utils::PushReadOnly(!node_selected);

        // Sigma slider -------------------------------------------------------
        if (this->interpolation_mode == TransferFunctionParam::InterpolationMode::GAUSS) {
            const float sigma_min = 0.0f;
            const float sigma_max = 2.0f;
            if (ImGui::SliderFloat("Selected Sigma", &this->widget_buffer.gauss_sigma, sigma_min, sigma_max)) {
                this->widget_buffer.gauss_sigma = std::clamp(this->widget_buffer.gauss_sigma, sigma_min, sigma_max);
                this->nodes[this->selected_node_index][5] = this->widget_buffer.gauss_sigma;
                this->reload_texture = true;
            }
            help = "[Ctrl + Left Click] for keyboard input";
            this->tooltip.Marker(help);
        }

        // Color editor for selected node -------------------------------------
        float edit_col[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (node_selected) {
            for (size_t i = 0; i < 4; i++) {
                edit_col[i] = this->nodes[this->selected_node_index][i];
            }
        }
        ImGuiColorEditFlags numberColorFlags =
            ImGuiColorEditFlags_DisplayRGB | ImGuiColorEditFlags_AlphaBar | ImGuiColorEditFlags_Float;
        if (ImGui::ColorEdit4("Selected Color", edit_col, numberColorFlags)) {
            this->nodes[this->selected_node_index][0] = edit_col[0];
            this->nodes[this->selected_node_index][1] = edit_col[1];
            this->nodes[this->selected_node_index][2] = edit_col[2];
            this->nodes[this->selected_node_index][3] = edit_col[3];
            this->reload_texture = true;
        }
        help = "[Left Click] on the colored square to open a color picker.\n"
               "[CTRL + Left Click] on individual component to input value.\n"
               "[Right Click] on the individual color widget to show options.";
        this->tooltip.Marker(help);

        // Value slider -------------------------------------------------------
        if (node_selected) {
            this->widget_buffer.range_value =
                (this->nodes[this->selected_node_index][4] * (this->range[1] - this->range[0])) + this->range[0];
        }
        if (ImGui::SliderFloat("Selected Value", &this->widget_buffer.range_value, this->range[0], this->range[1])) {
            this->widget_buffer.range_value =
                std::clamp(this->widget_buffer.range_value, this->range[0], this->range[1]);
            float new_x = (this->widget_buffer.range_value - this->range[0]) / (this->range[1] - this->range[0]);
            this->nodes[this->selected_node_index][4] = std::clamp(new_x, 0.0f, 1.0f);
            this->sortNodes(this->nodes, this->selected_node_index);
            this->reload_texture = true;
        }
        help = "[Ctrl + Left Click] for keyboard input";
        this->tooltip.Marker(help);

        // END selected NODE options ------------------------------------------
        megamol::gui::gui_utils::PopReadOnly(!node_selected);

        // Plot ---------------------------------------------------------------
        ImVec2 canvas_size = ImVec2(tfw_item_width, tfw_item_width / 2.0f);
        this->drawFunctionPlot(canvas_size);

        // Paint mode for plot
        megamol::gui::ButtonWidgets::ToggleButton("Paint Mode", this->plot_paint_mode);
        this->tooltip.Marker("[Left Drag] Create new nodes at mouse position.");

        // Color channels -----------------------------------------------------
        float available_width = tfw_item_width + style.ItemInnerSpacing.x + ImGui::GetScrollX();
        megamol::gui::ButtonWidgets::ToggleButton("Red", this->active_color_channels[0]);
        ImGui::SameLine();
        megamol::gui::ButtonWidgets::ToggleButton("Green", this->active_color_channels[1]);
        ImGui::SameLine();
        megamol::gui::ButtonWidgets::ToggleButton("Blue", this->active_color_channels[2]);
        ImGui::SameLine();
        megamol::gui::ButtonWidgets::ToggleButton("Alpha", this->active_color_channels[3]);
        ImGui::SameLine();
        ImGui::SameLine(available_width);
        ImGui::TextUnformatted("Color Channels");

        // Invert Colors
        if (ImGui::Button("All Nodes")) {
            for (auto& col : this->nodes) {
                for (int i = 0; i < 4; i++) {
                    col[i] = 1.0f - col[i];
                }
            }
            this->reload_texture = true;
        }
        ImGui::SameLine();
        megamol::gui::gui_utils::PushReadOnly(!node_selected);
        if (ImGui::Button("Selected Node")) {
            for (int i = 0; i < 4; i++) {
                this->nodes[this->selected_node_index][i] = 1.0f - this->nodes[this->selected_node_index][i];
            }
            this->reload_texture = true;
        }
        megamol::gui::gui_utils::PopReadOnly(!node_selected);
        ImGui::SameLine();
        ImGui::SameLine(tfw_item_width + style.ItemInnerSpacing.x + ImGui::GetScrollX());
        ImGui::TextUnformatted("Invert Colors");

        // Interpolation mode -------------------------------------------------
        std::map<TransferFunctionParam::InterpolationMode, std::string> opts;
        opts[TransferFunctionParam::InterpolationMode::LINEAR] = "Linear";
        opts[TransferFunctionParam::InterpolationMode::GAUSS] = "Gauss";
        const size_t opts_cnt = opts.size();
        if (ImGui::BeginCombo("Interpolation", opts[this->interpolation_mode].c_str())) {
            for (size_t i = 0; i < opts_cnt; ++i) {
                if (ImGui::Selectable(opts[(TransferFunctionParam::InterpolationMode) i].c_str(),
                        (this->interpolation_mode == (TransferFunctionParam::InterpolationMode) i))) {
                    this->interpolation_mode = (TransferFunctionParam::InterpolationMode) i;
                    this->reload_texture = true;
                }
            }
            ImGui::EndCombo();
        }

        // Presets -------------------------------------------------
        if (ImGui::BeginCombo("Load Preset", std::get<0>(PRESETS[0]).c_str())) {
            for (auto preset : PRESETS) {
                if (ImGui::Selectable(std::get<0>(preset).c_str())) {
                    std::get<1>(preset)(this->nodes, this->texture_size);
                    this->reload_texture = true;
                }
            }
            ImGui::EndCombo();
        }

        // Texture size -------------------------------------------------------
        ImGui::InputInt("Texture Size", &this->widget_buffer.tex_size, 1, 10, ImGuiInputTextFlags_None);
        if (ImGui::IsItemDeactivatedAfterEdit()) {
            this->texture_size = std::max(1, this->widget_buffer.tex_size);
            this->widget_buffer.tex_size = this->texture_size;
            this->reload_texture = true;
        }
    }
    // --------------------------------------------------------------------

    // Create current texture data
    if (this->reload_texture) {
        this->pending_changes = true;
        std::vector<float> texture_data;
        if (this->interpolation_mode == TransferFunctionParam::InterpolationMode::LINEAR) {
            texture_data =
                TransferFunctionParam::LinearInterpolation(static_cast<unsigned int>(this->texture_size), this->nodes);
        } else if (this->interpolation_mode == TransferFunctionParam::InterpolationMode::GAUSS) {
            texture_data =
                TransferFunctionParam::GaussInterpolation(static_cast<unsigned int>(this->texture_size), this->nodes);
        }
#ifdef MEGAMOL_USE_OPENGL
        if (!this->flip_legend) {
            this->image_widget_linear.LoadTextureFromData(this->texture_size, 1, texture_data.data());
            this->image_widget_nearest.LoadTextureFromData(
                this->texture_size, 1, texture_data.data(), GL_NEAREST, GL_NEAREST);
        } else {
            this->image_widget_linear.LoadTextureFromData(1, this->texture_size, texture_data.data());
            this->image_widget_nearest.LoadTextureFromData(
                1, this->texture_size, texture_data.data(), GL_NEAREST, GL_NEAREST);
        }
#else
        if (!this->flip_legend) {
            this->image_widget_linear.LoadTextureFromData(this->texture_size, 1, texture_data.data());
            this->image_widget_nearest.LoadTextureFromData(this->texture_size, 1, texture_data.data());
        } else {
            this->image_widget_linear.LoadTextureFromData(1, this->texture_size, texture_data.data());
            this->image_widget_nearest.LoadTextureFromData(1, this->texture_size, texture_data.data());
        }
#endif
        this->reload_texture = false;
    }

    // Apply -------------------------------------------------------
    bool apply_changes = false;
    if (this->show_options) {

        // Return true for current changes being applied
        gui_utils::PushReadOnly(!this->pending_changes);
        ImGui::PushStyleColor(
            ImGuiCol_Button, this->pending_changes ? GUI_COLOR_BUTTON_MODIFIED : style.Colors[ImGuiCol_Button]);
        ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
            this->pending_changes ? GUI_COLOR_BUTTON_MODIFIED_HIGHLIGHT : style.Colors[ImGuiCol_ButtonHovered]);
        ImGui::PushStyleColor(ImGuiCol_ButtonActive, style.Colors[ImGuiCol_ButtonActive]);

        if (ImGui::Button("Apply Pending Changes")) {
            apply_changes = true;
        }

        ImGui::PopStyleColor(3);
        gui_utils::PopReadOnly(!this->pending_changes);

        ImGui::SameLine();

        if (megamol::gui::ButtonWidgets::ToggleButton("Auto-apply", this->immediate_mode)) {
            apply_changes = this->immediate_mode;
        }

        if (this->immediate_mode && this->pending_changes) {
            apply_changes = true;
        }

        if (this->windowed_mode) {
            if (apply_changes) {
                if (this->IsAnyParameterConnected()) {
                    std::string tf;
                    if (this->GetTransferFunction(tf)) {
                        if (this->connected_parameter_ptr->Type() == ParamType_t::TRANSFERFUNCTION) {
                            this->connected_parameter_ptr->SetValue(tf);
                            this->connected_parameter_ptr->TransferFunctionEditor_SetHash(
                                this->connected_parameter_ptr->GetTransferFunctionHash());
                        }
                    }
                }
            }
        }
    }

    if (apply_changes) {
        this->pending_changes = false;
    }

    ImGui::PopItemWidth();

    ImGui::PopID();
    ImGui::EndGroup();

    return apply_changes;
}


void TransferFunctionEditor::PopUps() {

    // UNUSED
}


void TransferFunctionEditor::SpecificStateFromJSON(const nlohmann::json& in_json) {

    for (auto& header_item : in_json.items()) {
        if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
            for (auto& config_item : header_item.value().items()) {
                if (config_item.key() == this->Name()) {
                    auto config_values = config_item.value();

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"tfe_view_minimized"}, &this->win_view_minimized);
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"tfe_view_vertical"}, &this->win_view_vertical);
                    megamol::core::utility::get_json_value<std::string>(
                        config_values, {"tfe_active_param"}, &this->win_connected_param_name);
                    this->win_tfe_reset = true;
                }
            }
        }
    }
}


void TransferFunctionEditor::SpecificStateToJSON(nlohmann::json& inout_json) {

    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["tfe_view_minimized"] = this->win_view_minimized;
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["tfe_view_vertical"] = this->win_view_vertical;
    inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][this->Name()]["tfe_active_param"] = this->win_connected_param_name;
}


void TransferFunctionEditor::drawTextureBox(const ImVec2& size) {

    ImGuiStyle& style = ImGui::GetStyle();
    ImVec2 pos = ImGui::GetCursorScreenPos();
    if (texture_size == 0)
        return;

    ImVec2 image_size_interpol = size;
    ImVec2 image_size_nearest = ImVec2(size.x, size.y / 3.0f);
    if (this->flip_legend) {
        image_size_nearest = ImVec2(size.x / 3.0f, size.y);
    }

    // Use same texel offset as in shader
    float texel_min = (1.0f / static_cast<float>(this->texture_size)) * 0.5f;
    float texel_max = texel_min + 1.0f - (1.0f / static_cast<float>(this->texture_size));
    ImVec2 uv0 = ImVec2(texel_min, 0.0f);
    ImVec2 uv1 = ImVec2(texel_max, 1.0f);
    if (this->flip_legend) {
        uv0 = ImVec2(1.0f, texel_max);
        uv1 = ImVec2(0.0f, texel_min);
    }

    /// Nearest texel
    if (this->show_options) {
        if (!this->image_widget_nearest.IsLoaded()) {
            // Reserve layout space and draw a black background rectangle.
            ImDrawList* drawList = ImGui::GetWindowDrawList();
            ImGui::Dummy(image_size_nearest);
            drawList->AddRectFilled(pos, ImVec2(pos.x + image_size_nearest.x, pos.y + image_size_nearest.y),
                IM_COL32(0, 0, 0, 255), 0.0f, 10);
        } else {
            // Draw texture as image.
            this->image_widget_nearest.Widget(image_size_nearest, uv0, uv1);
        }
        if (this->flip_legend) {
            ImGui::SameLine(image_size_nearest.x + style.ItemInnerSpacing.x);
        }
    }

    /// Linear interpolated texel
    if (!this->image_widget_linear.IsLoaded()) {
        // Reserve layout space and draw a black background rectangle.
        ImDrawList* drawList = ImGui::GetWindowDrawList();
        ImGui::Dummy(image_size_interpol);
        drawList->AddRectFilled(pos, ImVec2(pos.x + image_size_interpol.x, pos.y + image_size_interpol.y),
            IM_COL32(0, 0, 0, 255), 0.0f, 10);
    } else {
        // Draw texture as image.
        this->image_widget_linear.Widget(image_size_interpol, uv0, uv1);
    }
    // Draw tooltip, if requested.
    if (ImGui::IsItemHovered()) {
        float xPx = ImGui::GetMousePos().x - pos.x - ImGui::GetScrollX();
        float xU = xPx / image_size_interpol.x;
        float xValue = xU * (this->range[1] - this->range[0]) + this->range[0];
        ImGui::BeginTooltip();
        ImGui::Text("%f Absolute Value\n%f Normalized Value", xValue, xU);
        ImGui::EndTooltip();
    }
}


void TransferFunctionEditor::drawScale(const ImVec2& pos, const ImVec2& size) {

    ImGuiStyle& style = ImGui::GetStyle();
    ImDrawList* drawList = ImGui::GetWindowDrawList();
    if (drawList == nullptr)
        return;
    ImVec2 reset_pos = ImGui::GetCursorScreenPos();

    const unsigned int scale_count = 3;

    float width = size.x;
    float height = size.y;
    float item_x_spacing = style.ItemInnerSpacing.x;
    float item_y_spacing = style.ItemInnerSpacing.y;

    // Draw scale lines
    const float line_length = (5.0f * megamol::gui::gui_scaling.Get());
    const float line_thickness = (2.0f * megamol::gui::gui_scaling.Get());
    const ImU32 line_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_Text]);

    ImVec2 init_pos = pos;
    float width_delta = 0.0f;
    float height_delta = 0.0f;
    if (this->flip_legend) {
        init_pos.x += width + item_x_spacing / 2.0f;
        if (this->show_options) { // when nearest texture is shown
            init_pos.x += width / 3.0f + item_x_spacing;
        }
        init_pos.y -= (height + item_y_spacing);
        height_delta = height / static_cast<float>(scale_count - 1);
    } else {
        init_pos.y -= item_y_spacing;
        width_delta = width / static_cast<float>(scale_count - 1);
    }

    for (unsigned int i = 0; i < scale_count; i++) {
        if (this->flip_legend) {
            float y = height_delta * static_cast<float>(i);
            if (i == 0)
                y += (line_thickness / 2.0f);
            if (i == (scale_count - 1))
                y -= (line_thickness / 2.0f);
            drawList->AddLine(
                init_pos + ImVec2(0.0f, y), init_pos + ImVec2(line_length, y), line_color, line_thickness);
        } else {
            float x = width_delta * static_cast<float>(i);
            if (i == 0)
                x += (line_thickness / 2.0f);
            if (i == (scale_count - 1))
                x -= (line_thickness / 2.0f);
            drawList->AddLine(
                init_pos + ImVec2(x, 0.0f), init_pos + ImVec2(x, line_length), line_color, line_thickness);
        }
    }

    // Draw scale text
    std::stringstream label_stream; /// String stream offers much better float formatting
    label_stream << this->range[0];
    std::string min_label_str = label_stream.str();
    float min_item_width = ImGui::CalcTextSize(min_label_str.c_str()).x;

    label_stream.str("");
    label_stream.clear();
    label_stream << this->range[1];
    std::string max_label_str = label_stream.str();
    float max_item_width = ImGui::CalcTextSize(max_label_str.c_str()).x;

    label_stream.str("");
    label_stream.clear();
    label_stream << (this->range[0] + ((this->range[1] - this->range[0]) / 2.0f));
    std::string mid_label_str = label_stream.str();
    float mid_item_width = ImGui::CalcTextSize(mid_label_str.c_str()).x;

    if (this->flip_legend) {
        float font_size = ImGui::GetFontSize();
        ImVec2 text_pos = init_pos + ImVec2(item_y_spacing + line_length, 0.0f);
        // Max Value
        ImGui::SetCursorScreenPos(text_pos);
        ImGui::TextUnformatted(max_label_str.c_str());
        // Middle Values
        float mid_value_height = (height - (2.0f * font_size) - (2.0f * item_y_spacing));
        if ((mid_value_height > font_size)) {
            ImGui::SetCursorScreenPos(text_pos + ImVec2(0.0f, (height / 2.0f) - (font_size / 2.0f)));
            ImGui::TextUnformatted(mid_label_str.c_str());
        }
        // Min Value
        ImGui::SetCursorScreenPos(text_pos + ImVec2(0.0f, (height - font_size)));
        ImGui::TextUnformatted(min_label_str.c_str());
    } else {
        ImGui::SetCursorScreenPos(pos + ImVec2(0.0f, line_length));
        // Min Value
        ImGui::TextUnformatted(min_label_str.c_str());
        // Middle Values
        float mid_value_width = (width - min_item_width - max_item_width - (2.0f * item_x_spacing));
        if ((mid_value_width > mid_item_width)) {
            ImGui::SameLine((width / 2.0f) - (mid_item_width / 2.0f) + ImGui::GetScrollX());
            ImGui::TextUnformatted(mid_label_str.c_str());
        }
        // Max Value
        ImGui::SameLine(width - max_item_width + ImGui::GetScrollX());
        ImGui::TextUnformatted(max_label_str.c_str());
    }

    if (this->flip_legend) {
        ImGui::SetCursorScreenPos(reset_pos);
    }
}


void TransferFunctionEditor::drawFunctionPlot(const ImVec2& size) {
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    ImDrawList* drawList = ImGui::GetWindowDrawList();
    assert(drawList != nullptr);

    ImVec2 canvas_pos = ImGui::GetCursorScreenPos(); // ImDrawList API uses screen coordinates!
    ImVec2 canvas_size = size;
    if (canvas_size.x < 100.0f)
        canvas_size.x = 100.0f;
    if (canvas_size.y < 100.0f)
        canvas_size.y = 100.0f;
    ImVec2 mouse_pos = ImGui::GetMousePos(); // current mouse position

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

    const float line_width = (2.0f * megamol::gui::gui_scaling.Get());
    const float point_radius = (6.0f * megamol::gui::gui_scaling.Get());
    const float point_border_width = (2.0f * megamol::gui::gui_scaling.Get());
    const float point_border_radius = point_radius + point_border_width;

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

    unsigned int current_selected_node_index = GUI_INVALID_ID;
    unsigned int current_selected_channel_index = GUI_INVALID_ID;
    ImVec2 current_selected_node_drag_delta = ImVec2(0.0f, 0.0f);
    float dist_delta = FLT_MAX;

    // Draw line for selected node
    bool node_selected =
        ((this->selected_node_index != GUI_INVALID_ID) && (this->selected_node_index < this->nodes.size()));
    if (node_selected) {
        drawList->AddLine(
            ImVec2((canvas_pos.x + this->nodes[this->selected_node_index][4] * canvas_size.x), canvas_pos.y),
            ImVec2((canvas_pos.x + this->nodes[this->selected_node_index][4] * canvas_size.x),
                (canvas_pos.y + canvas_size.y)),
            ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_TextDisabled]), point_border_width);
    }

    // For each enabled color channel
    for (unsigned int c = 0; c < channelColors.size(); ++c) {
        if (!this->active_color_channels[c]) {
            continue;
        }
        const auto node_count = this->nodes.size();

        // Draw lines.
        drawList->PathClear();
        for (unsigned int i = 0; i < node_count; ++i) {

            ImVec2 point = ImVec2(canvas_pos.x + this->nodes[i][4] * canvas_size.x,
                canvas_pos.y + (1.0f - this->nodes[i][c]) * canvas_size.y);
            if (this->interpolation_mode == TransferFunctionParam::InterpolationMode::LINEAR) {
                drawList->PathLineTo(point);
            } else if (this->interpolation_mode == TransferFunctionParam::InterpolationMode::GAUSS) {
                const float ga = this->nodes[i][c];
                const float gb = this->nodes[i][4];
                const float gc = this->nodes[i][5];
                const int step = 3; // pixel step width in x direction
                float x, g;
                ImVec2 pos;

                for (int p = 0; p < (int) canvas_size.x + step; p += step) {
                    x = (float) (p) / canvas_size.x;
                    g = TransferFunctionParam::gauss(x, ga, gb, gc);
                    pos =
                        ImVec2(canvas_pos.x + (x * canvas_size.x), canvas_pos.y + canvas_size.y - (g * canvas_size.y));
                    drawList->PathLineTo(pos);
                }
                drawList->PathStroke(channelColors[c], ImDrawFlags_None, line_width);
            }

            // Test for intersection of mouse position with node circle
            ImVec2 mouse_delta = ImVec2(point.x - mouse_pos.x, point.y - mouse_pos.y);
            auto dist = glm::length(glm::vec2(mouse_delta.x, mouse_delta.y));
            // Select node with minimized mouse delta
            if ((dist <= point_border_radius) && (dist < dist_delta)) {
                current_selected_node_index = i;
                current_selected_channel_index = c;
                current_selected_node_drag_delta = mouse_delta;
                dist_delta = dist;
            }
        }
        drawList->PathStroke(channelColors[c], ImDrawFlags_None, line_width);

        // Draw node circles.
        for (unsigned int i = 0; i < node_count; ++i) {

            if ((((i != current_selected_node_index) || (c != current_selected_channel_index)) &&
                    (i != this->selected_node_index)) &&
                /*1*/ //((point_radius * 1.25f) > (canvas_size.x / static_cast<float>(this->texture_size)))) {
                /*2*/ ((static_cast<float>(node_count) * point_radius * 1.25f) > canvas_size.x)) {
                // Only draw hovered circle *1* if nodes are larger than texel size or *2* if there are too many nodes
                continue;
            }

            // Draw node point
            ImVec2 point = ImVec2(canvas_pos.x + this->nodes[i][4] * canvas_size.x,
                canvas_pos.y + (1.0f - this->nodes[i][c]) * canvas_size.y);
            if (i == this->selected_node_index) {
                auto selected_circle_color = ImGui::ColorConvertFloat4ToU32(style.Colors[ImGuiCol_ButtonActive]);
                drawList->AddCircleFilled(point, point_border_radius, selected_circle_color);
            }
            auto point_color =
                ImGui::ColorConvertFloat4ToU32(ImVec4(this->nodes[i][0], this->nodes[i][1], this->nodes[i][2], 1.0f));
            drawList->AddCircleFilled(point, point_radius, point_color);
        }
    }
    drawList->PopClipRect();

    // Process plot interaction -----------------------------------------------
    ImGui::InvisibleButton("plot", canvas_size); // Needs to catch mouse input
    if (ImGui::IsItemHovered()) {
        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) { // Left Mouse Click
            this->changeNodeSelection(
                current_selected_node_index, current_selected_channel_index, current_selected_node_drag_delta);
            if (this->plot_paint_mode) {
                this->selected_node_drag_delta = mouse_pos;
            }
        } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) { // Left Mouse Drag
            this->plot_dragging = true;
        } else if (ImGui::IsMouseClicked(ImGuiMouseButton_Right)) { // Right Mouse Click
            if (!this->deleteNode(current_selected_node_index)) {
                this->addNode(mouse_pos, canvas_pos, canvas_size);
            }
        }
    }
    // Track mouse even outside canvas in paint mode when dragging started within canvas
    if (this->plot_dragging && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
        if (!this->plot_paint_mode) {
            this->moveSelectedNode(mouse_pos, canvas_pos, canvas_size);
        } else {
            this->paintModeNode(mouse_pos, canvas_pos, canvas_size);
        }
    } else {
        this->plot_dragging = false;
    }
    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted("Function Plot");
    this->tooltip.Marker(
        "[Left Click] Select Node\n[Left Drag] Move Node\n[Right Click] Add Node/Delete Selected Node");
}


bool TransferFunctionEditor::addNode(const ImVec2& mouse_pos, const ImVec2& canvas_pos, const ImVec2& canvas_size) {

    float new_x = (mouse_pos.x - canvas_pos.x) / canvas_size.x;
    new_x = std::clamp(new_x, 0.0f, 1.0f);

    float new_y = 1.0f - ((mouse_pos.y - canvas_pos.y) / canvas_size.y);
    new_y = std::max(0.0f, std::min(new_y, 1.0f));

    TransferFunctionParam::NodeData_t new_node = {0.0f, 0.0f, 0.0f, 0.0f, new_x, 0.05f};
    for (unsigned int cc = 0; cc < 4; cc++) {
        if (this->active_color_channels[cc]) {
            new_node[cc] = new_y;
        }
    }

    if (this->nodes.empty()) {
        this->nodes.push_back(new_node);
    } else {
        bool added_node = false;
        for (auto it = this->nodes.begin(); it != this->nodes.end(); ++it) {
            new_node = (*it);
            for (unsigned int cc = 0; cc < 4; cc++) {
                if (this->active_color_channels[cc]) {
                    new_node[cc] = new_y;
                }
            }
            new_node[4] = new_x;
            new_node[5] = 0.05f;

            if (new_x < (*it)[4]) {
                if (it != this->nodes.begin()) {
                    auto pre_col = (*(it - 1));
                    auto post_col = (*it);
                    new_node = {(pre_col[0] + post_col[0]) / 2.0f, (pre_col[1] + post_col[1]) / 2.0f,
                        (pre_col[2] + post_col[2]) / 2.0f, (pre_col[3] + post_col[3]) / 2.0f, new_x, 0.05f};
                    for (unsigned int cc = 0; cc < 4; cc++) {
                        if (this->active_color_channels[cc]) {
                            new_node[cc] = new_y;
                        }
                    }
                }
                this->nodes.insert(it, new_node);
                added_node = true;
                break;
            }
        }
        if (!added_node) {
            this->nodes.push_back(new_node);
        }
    }

    this->sortNodes(this->nodes, this->selected_node_index);
    this->reload_texture = true;
    return true;
}


bool TransferFunctionEditor::paintModeNode(
    const ImVec2& mouse_pos, const ImVec2& canvas_pos, const ImVec2& canvas_size) {

    this->selected_node_index = GUI_INVALID_ID;

    /// Create new nodes depending on texture size - overwrite/delete existing nodes
    float x = (mouse_pos.x - canvas_pos.x) / canvas_size.x;
    float texel_delta = 1.0f / static_cast<float>(this->texture_size);
    float texel_cnt = floorf(x / texel_delta);
    float pre_texel = texel_delta * texel_cnt;
    float post_texel = texel_delta * (texel_cnt + 1.0f);
    texel_cnt = ((pre_texel - x) > (x - post_texel)) ? (texel_cnt) : (texel_cnt + 1.0f);
    float new_x = std::clamp((texel_delta * texel_cnt), 0.0f, 1.0f);

    std::vector<unsigned int> delete_nodes_indices;
    unsigned int node_count = this->nodes.size();
    float tmp_value;
    float delta_min_x = (new_x - texel_delta);
    float drag_delta_x = (this->selected_node_drag_delta.x - canvas_pos.x) / canvas_size.x;
    if (delta_min_x > drag_delta_x) {
        delta_min_x = drag_delta_x;
    }
    delta_min_x += TF_FLOAT_EPS;
    float delta_max_x = (new_x + texel_delta) - TF_FLOAT_EPS;
    if (delta_max_x < drag_delta_x) {
        delta_max_x = drag_delta_x;
    }
    delta_max_x -= TF_FLOAT_EPS;
    for (unsigned int i = 0; i < node_count; i++) {
        tmp_value = this->nodes[i][4];
        if ((tmp_value > delta_min_x) && (tmp_value < delta_max_x)) {
            delete_nodes_indices.push_back(i);
        }
    }
    // Reverse erase items to keep indices valid while erasing
    for (auto i = delete_nodes_indices.rbegin(); i != delete_nodes_indices.rend(); ++i) {
        this->nodes.erase(this->nodes.begin() + (*i));
    }

    float new_mouse_pos_x = (new_x * canvas_size.x) + canvas_pos.x;
    ImVec2 new_pos = ImVec2(new_mouse_pos_x, mouse_pos.y);
    this->addNode(new_pos, canvas_pos, canvas_size);
    this->selected_node_drag_delta = new_pos;

    return true;
}


bool TransferFunctionEditor::changeNodeSelection(unsigned int new_selected_node_index,
    unsigned int new_selected_channel_index, ImVec2 new_selected_node_drag_delta) {

    this->selected_node_index = new_selected_node_index;
    this->selected_channel_index = new_selected_channel_index;
    this->selected_node_drag_delta = new_selected_node_drag_delta;

    if ((this->selected_node_index != GUI_INVALID_ID) && (this->selected_node_index < this->nodes.size())) {
        this->widget_buffer.range_value =
            (this->nodes[this->selected_node_index][4] * (this->range[1] - this->range[0])) + this->range[0];
        this->widget_buffer.gauss_sigma = this->nodes[this->selected_node_index][5];
    } else {
        this->widget_buffer.range_value = this->widget_buffer.left_range;
        this->widget_buffer.gauss_sigma = 0.05f;
    }
    return true;
}


bool TransferFunctionEditor::moveSelectedNode(
    const ImVec2& mouse_pos, const ImVec2& canvas_pos, const ImVec2& canvas_size) {

    if ((this->selected_node_index != GUI_INVALID_ID) && (this->selected_node_index < this->nodes.size())) {

        float new_x = (mouse_pos.x - canvas_pos.x + this->selected_node_drag_delta.x) / canvas_size.x;
        new_x = std::clamp(new_x, 0.0f, 1.0f);

        float new_y = 1.0f - ((mouse_pos.y - canvas_pos.y + this->selected_node_drag_delta.y) / canvas_size.y);
        new_y = std::max(0.0f, std::min(new_y, 1.0f));

        this->nodes[this->selected_node_index][4] = new_x;
        this->widget_buffer.range_value = (new_x * (this->range[1] - this->range[0])) + this->range[0];

        for (unsigned int cc = 0; cc < 4; cc++) {
            if (this->active_color_channels[cc] && (this->selected_channel_index == cc)) {
                this->nodes[this->selected_node_index][cc] = new_y;
            }
        }

        this->sortNodes(this->nodes, this->selected_node_index);
        this->reload_texture = true;
        return true;
    }
    return false;
}


bool TransferFunctionEditor::deleteNode(unsigned int node_index) {

    if ((node_index != GUI_INVALID_ID) && (node_index < this->nodes.size())) {

        this->nodes.erase(this->nodes.begin() + node_index);
        this->selected_node_index = GUI_INVALID_ID;

        this->sortNodes(this->nodes, this->selected_node_index);
        this->reload_texture = true;
        return true;
    }
    return false;
}


void TransferFunctionEditor::sortNodes(TransferFunctionParam::NodeVector_t& n, unsigned int& selected_node_idx) const {

    // Save current value of selected node
    auto n_count = static_cast<unsigned int>(n.size());
    float value = 0.0f;
    if (this->selected_node_index < n_count) {
        value = n[this->selected_node_index][4];
    }

    // Sort nodes by value
    std::sort(n.begin(), n.end(),
        [](const TransferFunctionParam::NodeData_t& nd1, const TransferFunctionParam::NodeData_t& nd2) {
            return (nd1[4] < nd2[4]);
        });

    // Prevent nodes with same value
    for (int i = 0; i < (static_cast<int>(n_count) - 1); i++) {
        if (n[i][4] == n[i + 1][4]) {
            if (value == 0.0f) {
                n[i][4] += TF_FLOAT_EPS;
            } else if (value == 1.0f) {
                n[i + 1][4] -= TF_FLOAT_EPS;
            } else {
                n[i + 1][4] += TF_FLOAT_EPS;
            }
        }
    }

    // Search for value of last selected node
    if (this->selected_node_index < n_count) {
        for (unsigned int i = 0; i < n_count; i++) {
            if (n[i][4] == value) {
                selected_node_idx = i;
                break;
            }
        }
    }
}

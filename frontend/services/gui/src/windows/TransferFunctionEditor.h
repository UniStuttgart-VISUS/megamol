/*
 * TransferFunctionEditor.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_INCLUDED
#define MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_INCLUDED
#pragma once


#include "AbstractWindow.h"
#include "mmcore/param/TransferFunctionParam.h"
#include "widgets/HoverToolTip.h"
#include "widgets/ImageWidget.h"

using namespace megamol::core::param;


namespace megamol {
namespace gui {


// Forward declarations
class Parameter;

/** ************************************************************************
 * 1D Transfer Function Editor GUI window
 */
class TransferFunctionEditor : public AbstractWindow {
public:
    TransferFunctionEditor(const std::string& window_name, bool windowed);
    ~TransferFunctionEditor() = default;

    bool Update() override;
    bool Draw() override;
    void PopUps() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

    /**
     * Set transfer function data to use in editor.
     *
     * @param tfs The transfer function encoded as string in JSON format.
     *
     * @return True if string was successfully converted into transfer function data, false otherwise.
     */
    void SetTransferFunction(const std::string& tfs, bool connected_parameter_mode, bool full_init);

    /**
     * Get current transfer function data.
     *
     * @return The transfer function encoded as string in JSON format
     */
    bool GetTransferFunction(std::string& tfs);

    /**
     * Set the currently connected parameter.
     */
    void SetConnectedParameter(Parameter* param_ptr, const std::string& param_full_name);

    /**
     * Returns true if editor is in minimized view.
     */
    inline bool IsMinimized() const {
        return !this->show_options;
    }

    /**
     * Set minimized view.
     */
    inline void SetMinimized(bool minimized) {
        this->show_options = !minimized;
    }

    /**
     * Returns true if editor is in vertical view.
     */
    inline bool IsVertical() const {
        return this->flip_legend;
    }

    /**
     * Set vertical view.
     */
    inline void SetVertical(bool vertical) {
        this->flip_legend = vertical;
    }

    bool IsParameterConnected() const {
        return (this->connected_parameter_ptr != nullptr);
    }

    std::string ProcessParameterConnectionRequest() {
        auto rpnc = this->request_parameter_name_connect;
        this->request_parameter_name_connect.clear();
        return rpnc;
    }

private:
    /** The global input widget state buffer. */
    struct WidgetBuffer {
        float left_range;
        float right_range;
        float gauss_sigma;
        float range_value;
        int tex_size;
    };

    // VARIABLES -----------------------------------------------------------

    const bool windowed_mode;
    /** The currently active parameter whose transfer function is currently loaded into this editor. */
    Parameter* connected_parameter_ptr;
    TransferFunctionParam::NodeVector_t nodes;
    std::array<float, 2> range;
    std::array<float, 2> last_range;
    bool range_overwrite;
    TransferFunctionParam::InterpolationMode interpolation_mode;
    bool reload_texture;
    int texture_size;
    bool pending_changes;
    bool immediate_mode;
    std::array<bool, 4> active_color_channels;
    unsigned int selected_channel_index;
    unsigned int selected_node_index;
    ImVec2 selected_node_drag_delta;
    bool show_options;
    WidgetBuffer widget_buffer;
    bool flip_legend;
    bool check_once_force_set_overwrite_range;
    bool plot_paint_mode;
    bool plot_dragging;
    std::string request_parameter_name_connect;

    bool win_view_minimized;              // [SAVED] flag indicating minimized window state
    bool win_view_vertical;               // [SAVED] flag indicating vertical window state
    std::string win_connected_param_name; // [SAVED] last active parameter connected to editor
    bool win_tfe_reset;                   // flag for reset of tfe window on state loading

    // Widgets
    HoverToolTip tooltip;
    ImageWidget image_widget_linear;
    ImageWidget image_widget_nearest;

    // FUNCTIONS -----------------------------------------------------------

    void drawTextureBox(const ImVec2& size);
    void drawScale(const ImVec2& pos, const ImVec2& size);
    void drawFunctionPlot(const ImVec2& size);

    bool addNode(const ImVec2& mouse_pos, const ImVec2& canvas_pos, const ImVec2& canvas_size);
    bool paintModeNode(const ImVec2& mouse_pos, const ImVec2& canvas_pos, const ImVec2& canvas_size);
    bool changeNodeSelection(unsigned int new_selected_node_index, unsigned int new_selected_channel_index,
        ImVec2 new_selected_node_drag_delta);
    bool moveSelectedNode(const ImVec2& mouse_pos, const ImVec2& canvas_pos, const ImVec2& canvas_size);
    bool deleteNode(unsigned int node_index);
    void sortNodes(TransferFunctionParam::NodeVector_t& n, unsigned int& selected_node_idx) const;
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_TRANSFERFUNCTIONEDITOR_INCLUDED

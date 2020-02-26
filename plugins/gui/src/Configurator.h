/*
 * Configurator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
#define MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED


#include "mmcore/CoreInstance.h"
#include "mmcore/view//Input.h"

#include "vislib/sys/Log.h"

#include <map>
#include <math.h> // fmodf
#include <tuple>

#include "FileUtils.h"
#include "GUIUtils.h"
#include "WindowManager.h"
#include "graph/GraphManager.h"


namespace megamol {
namespace gui {

class Configurator {
public:
    /**
     * CTOR.
     */
    Configurator();

    /**
     * DTOR.
     */
    virtual ~Configurator();

    /**
     * Draw configurator window.
     */
    bool Draw(WindowManager::WindowConfiguration& wc, megamol::core::CoreInstance* core_instance);

    /**
     * Checks if any hotkeys are pressed.
     *
     * @return true when any hotkey is pressed.
     */
    bool CheckHotkeys(void);

    /*
     * Provide additional font for independent scaling of font used in graph.
     */
    inline void SetGraphFont(ImFont* graph_font) { this->gui.graph_font = graph_font; }

private:
    // VARIABLES --------------------------------------------------------------

    typedef std::tuple<megamol::core::view::KeyCode, bool> HotkeyData;
    enum HotkeyIndex : size_t { MODULE_SEARCH = 0, PARAMETER_SEARCH = 1, DELETE_GRAPH_ITEM = 2, INDEX_COUNT = 3 };

    std::array<HotkeyData, HotkeyIndex::INDEX_COUNT> hotkeys;

    graph::GraphManager graph_manager;
    GUIUtils utils;

    struct Gui {
        int window_state;
        std::string project_filename;
        graph::GraphManager::GraphPtrType graph_ptr;
        int selected_list_module_id;
        bool rename_popup_open;
        std::string* rename_popup_string;
        float mouse_wheel;
        ImFont* graph_font;
        bool update_current_graph;
        float split_width_left;
        float split_width_right;
    } gui;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu(megamol::core::CoreInstance* core_instance);
    void draw_window_module_list(float width);
    void draw_window_graph(float width);
    void draw_window_parameter_list(float width);

    bool draw_graph_menu(graph::GraphManager::GraphPtrType graph);
    bool draw_graph_canvas(graph::GraphManager::GraphPtrType graph);
    bool draw_canvas_grid(graph::GraphManager::GraphPtrType graph);
    bool draw_canvas_calls(graph::GraphManager::GraphPtrType graph);
    bool draw_canvas_modules(graph::GraphManager::GraphPtrType graph);
    bool draw_canvas_module_call_slots(graph::GraphManager::GraphPtrType graph, graph::Graph::ModuleGraphPtrType mod);
    bool draw_canvas_dragged_call(graph::GraphManager::GraphPtrType graph);

    bool update_module_size(graph::GraphManager::GraphPtrType graph, graph::Graph::ModuleGraphPtrType mod);
    bool update_slot_position(graph::GraphManager::GraphPtrType graph, graph::Graph::CallSlotGraphPtrType slot);
    bool update_graph_layout(graph::GraphManager::GraphPtrType graph);

    bool add_new_module_to_graph(
        const graph::Graph::StockModule& mod, int compat_call_idx, const std::string& compat_call_slot_name);

    bool popup_save_project(bool open, megamol::core::CoreInstance* core_instance);

    inline const std::string get_unique_project_name(void) {
        return ("Project_" + std::to_string(this->graph_manager.GetGraphs().size() + 1));
    }
    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
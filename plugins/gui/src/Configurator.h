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
#include "GraphManager.h"
#include "WindowManager.h"


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
    enum HotkeyIndex : size_t { MODULE_SEARCH = 0, DELETE_GRAPH_ITEM = 1, INDEX_COUNT = 2 };

    std::array<HotkeyData, HotkeyIndex::INDEX_COUNT> hotkeys;

    GraphManager graph_manager;
    GUIUtils utils;

    struct Gui {
        int window_state;
        std::string project_filename;
        GraphManager::GraphPtrType graph_ptr;
        int selected_list_module_id;
        bool rename_popup_open;
        std::string* rename_popup_string;
        float mouse_wheel;
        ImFont* graph_font;
        bool update_current_graph;
        float split_thickness;
        float split_width_left;
        float split_width_right;
    } gui;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu(megamol::core::CoreInstance* core_instance);
    void draw_window_module_list(float width);
    void draw_window_graph(float width);
    void draw_window_parameter_list(float width);

    bool draw_graph_menu(GraphManager::GraphPtrType graph);
    bool draw_graph_canvas(GraphManager::GraphPtrType graph);
    bool draw_graph_grid(GraphManager::GraphPtrType graph);
    bool draw_graph_calls(GraphManager::GraphPtrType graph);
    bool draw_graph_modules(GraphManager::GraphPtrType graph);
    bool draw_graph_module_call_slots(GraphManager::GraphPtrType graph, Graph::ModulePtrType mod);
    bool draw_graph_dragged_call(GraphManager::GraphPtrType graph);

    bool update_module_size(GraphManager::GraphPtrType graph, Graph::ModulePtrType mod);
    bool update_slot_position(GraphManager::GraphPtrType graph, Graph::CallSlotPtrType slot);
    bool update_graph_layout(GraphManager::GraphPtrType graph);

    bool add_new_module_to_graph(
        const Graph::StockModule& mod, int compat_call_idx, const std::string& compat_call_slot_name);

    bool popup_save_project(bool open, megamol::core::CoreInstance* core_instance);

    inline const std::string get_unique_project_name(void) {
        return ("Project_" + std::to_string(this->graph_manager.GetGraphs().size() + 1));
    }
    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
/*
 * Configurator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */
// Creating a node graph editor for ImGui
// Quick demo, not production code! This is more of a demo of how to use ImGui to create custom stuff.
// Better version by @daniel_collin here https://gist.github.com/emoon/b8ff4b4ce4f1b43e79f2
// See https://github.com/ocornut/imgui/issues/306
// v0.03: fixed grid offset issue, inverted sign of 'scrolling'
// Animated gif: https://cloud.githubusercontent.com/assets/8225057/9472357/c0263c04-4b4c-11e5-9fdf-2cd4f33f6582.gif

#ifndef MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
#define MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED

#include <imgui.h>
#define IMGUI_DEFINE_MATH_OPERATORS
#include <imgui_internal.h>
#include "imgui_impl_opengl3.h"
#include "imgui_stdlib.h"

#include <map>
#include <math.h> // fmodf
#include <tuple>

#include "mmcore/CoreInstance.h"
#include "mmcore/view//Input.h"

#include "FileUtils.h"
#include "GUIUtils.h"
#include "GraphManager.h"
#include "WindowManager.h"

#include "vislib/sys/Log.h"


namespace megamol {
namespace gui {

class Configurator {
public:
    /**
     * Initialises a new instance.
     */
    Configurator();

    /**
     * Finalises an instance.
     */
    virtual ~Configurator();

    /**
     * Draw configurator ImGui window.
     * (Call in GUIView::drawConfiguratorCallback())
     */
    bool Draw(WindowManager::WindowConfiguration& wc, megamol::core::CoreInstance* core_instance);

    /**
     * Checks if any hotkeys are pressed.
     * (Call in GUIView::OnKey())
     *
     * @return true when any hotkey is pressed.
     */
    bool CheckHotkeys(void);

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
    } gui;

    // FUNCTIONS --------------------------------------------------------------

    bool draw_window_menu(megamol::core::CoreInstance* core_instance);
    bool draw_window_module_list(void);

    bool draw_canvas_graph(GraphManager::GraphPtrType graph);
    bool draw_canvas_grid(GraphManager::GraphPtrType graph);
    bool draw_canvas_calls(GraphManager::GraphPtrType graph);
    bool draw_canvas_modules(GraphManager::GraphPtrType graph);
    bool draw_canvas_module_call_slots(GraphManager::GraphPtrType graph, Graph::ModulePtrType mod);
    bool draw_canvas_dragged_call(GraphManager::GraphPtrType graph);

    bool update_module_size(GraphManager::GraphPtrType graph, Graph::ModulePtrType mod);
    bool layout_graph(GraphManager::GraphPtrType graph);

    bool popup_save_project(bool open, megamol::core::CoreInstance* core_instance);

    bool add_new_graph(void);
    bool add_new_module_to_graph(
        Graph::StockModule& mod, int compat_call_idx, const std::string& compat_call_slot_name);

    // ------------------------------------------------------------------------
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
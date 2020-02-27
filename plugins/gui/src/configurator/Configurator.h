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
namespace configurator {

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

    GraphManager graph_manager;
    GUIUtils utils;

    struct Gui {
        int window_state;
        std::string project_filename;
        GraphManager::GraphPtrType graph_ptr;
        int selected_list_module_id;
        ImFont* graph_font;
        float split_width;
    } gui;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu(megamol::core::CoreInstance* core_instance);
    void draw_window_module_list(float width);

    // bool add_new_module_to_graph(const StockModule& mod, int compat_call_idx, const std::string&
    // compat_call_slot_name);

    bool popup_save_project(bool open, megamol::core::CoreInstance* core_instance);

    inline const std::string get_unique_project_name(void) {
        return ("Project_" + std::to_string(this->graph_manager.GetGraphs().size() + 1));
    }
    // ------------------------------------------------------------------------
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
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

#include "FileUtils.h"
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
    inline void SetGraphFont(ImFont* graph_font) { this->graph_font = graph_font; }

private:
    // VARIABLES --------------------------------------------------------------
    enum HotkeyIndex : size_t { MODULE_SEARCH = 0, PARAMETER_SEARCH = 1, DELETE_GRAPH_ITEM = 2, INDEX_COUNT = 3 };
    std::array<HotkeyData, HotkeyIndex::INDEX_COUNT> hotkeys;

    GraphManager graph_manager;
    GUIUtils utils;

    int window_state;
    std::string project_filename;
    GraphManager::GraphPtrType graph_ptr;
    int selected_list_module_id;
    ImFont* graph_font;
    float split_width;

    unsigned int unique_project_id;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu(megamol::core::CoreInstance* core_instance);
    void draw_window_module_list(float width);

    bool popup_save_project(bool open, megamol::core::CoreInstance* core_instance);

    inline const std::string get_unique_project_name(void) {
        return ("Project_" + std::to_string(++unique_project_id));
    }
    // ------------------------------------------------------------------------
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
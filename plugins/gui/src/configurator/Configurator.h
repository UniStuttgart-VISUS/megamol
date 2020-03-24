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

    HotKeyArrayType hotkeys;

    GraphManager graph_manager;
    megamol::gui::FileUtils file_utils;    
    megamol::gui::GUIUtils utils;

    int window_state;
    std::string project_filename;
    ImGuiID graph_uid;
    ImGuiID selected_list_module_uid;
    ImFont* graph_font;
    float child_split_width;
    ImGuiID add_project_graph_uid;

    unsigned int project_uid;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu(megamol::core::CoreInstance* core_instance);

    void draw_window_module_list(float width);

    void add_empty_project(void);

    inline const std::string get_unique_project_name(void) { return ("Project_" + std::to_string(++project_uid)); }
    // ------------------------------------------------------------------------
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
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
    inline void SetGraphFont(ImFont* graph_font) { this->state.font = graph_font; }

private:
    // VARIABLES --------------------------------------------------------------

    GraphManager graph_manager;
    megamol::gui::FileUtils file_utils;
    megamol::gui::GUIUtils utils;

    int init_state;
    float left_child_width;
    ImGuiID selected_list_module_uid;
    ImGuiID add_project_graph_uid;
    bool show_module_list_sidebar;
    bool show_module_list_child;
    ImVec2 module_list_popup_pos;
    ImGuiID last_selected_callslot_uid;
    std::string project_filename;

    megamol::gui::GraphStateType state;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu(megamol::core::CoreInstance* core_instance);

    void draw_window_module_list(float width);

    void add_empty_project(void);
};

} // namespace configurator
} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED

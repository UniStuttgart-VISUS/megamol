/*
 * Configurator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
#define MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED


#include "WindowCollection.h"
#include "graph/GraphCollection.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/SplitterWidget.h"
#include "widgets/StringSearchWidget.h"

#include "mmcore/CoreInstance.h"
#include "mmcore/param/ParamSlot.h"
#include "mmcore/param/StringParam.h"

// Used for platform independent clipboard (ImGui so far only provides windows implementation)
#ifdef GUI_USE_GLFW
#    include "GLFW/glfw3.h"
#endif


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
     * Get hotkey of configurator.
     *
     * @return Hotkeys of configurator.
     */
    inline megamol::gui::HotkeyArray_t& GetHotkeys(void) { return this->graph_state.hotkeys; }

    /**
     * Draw configurator window.
     */
    bool Draw(WindowCollection::WindowConfiguration& wc, megamol::core::CoreInstance* core_instance);

    /**
     * Returns required font scalings for graph canvas
     */
    inline const FontScalingArray_t& GetGraphFontScalings(void) const { return this->graph_state.font_scalings; }

    /**
     * Return list of parameter slots provided by this class. Make available in module which uses this class.
     */
    inline const std::vector<megamol::core::param::ParamSlot*> GetParams(void) const { return this->param_slots; }

    /**
     * Save current configurator state to state parameter.
     */
    void UpdateStateParameter(void);

    /**
     * Return graph collection.
     */
    GraphCollection& GetGraphCollection(void) { return this->graph_collection; }

private:
    // VARIABLES --------------------------------------------------------------

    GraphCollection graph_collection;

    std::vector<megamol::core::param::ParamSlot*> param_slots;
    megamol::core::param::ParamSlot state_param;

    static std::vector<std::string> dropped_files;

    int init_state;
    float module_list_sidebar_width;
    ImGuiID selected_list_module_uid;
    ImGuiID add_project_graph_uid;
    ImGuiID module_list_popup_hovered_group_uid;
    bool show_module_list_sidebar;
    bool show_module_list_child;
    ImVec2 module_list_popup_pos;
    bool module_list_popup_hovered;
    ImGuiID last_selected_callslot_uid;
    megamol::gui::GraphState_t graph_state;
    bool open_popup_load;

    // Widgets
    FileBrowserWidget file_browser;
    StringSearchWidget search_widget;
    SplitterWidget splitter_widget;
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu(megamol::core::CoreInstance* core_instance);
    void draw_window_module_list(float width);

    void add_empty_project(void);

    bool configurator_state_from_json_string(const std::string& json_string);
    bool configurator_state_to_json(nlohmann::json& out_json);

    void drawPopUps(void);

#ifdef GUI_USE_GLFW
    /**
     * NB: Successfully testet using Windows10 and (X)Ubuntu with "Nautilus" file browser as drag source of the files.
     *     Failed using (X)Ubuntu with "Thunar" file browser.
     *     GLFW: File drop is currently unimplemented for "Wayland" (e.g. Fedora using GNOME)
     */
    static void file_drop_callback(::GLFWwindow* window, int count, const char* paths[]);
#endif
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED

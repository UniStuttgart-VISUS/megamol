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
        inline megamol::gui::HotkeyArray_t& GetHotkeys(void) {
            return this->graph_state.hotkeys;
        }

        /**
         * Draw configurator window.
         */
        bool Draw(WindowCollection::WindowConfiguration& wc);

        /**
         * Returns required font scalings for graph canvas
         */
        inline const FontScalingArray_t& GetGraphFontScalings(void) const {
            return this->graph_state.graph_zoom_font_scalings;
        }

        /**
         * Return graph collection.
         */
        GraphCollection& GetGraphCollection(void) {
            return this->graph_collection;
        }
        /**
         * State to and from JSON.
         */
        bool StateToJSON(nlohmann::json& inout_json);
        bool StateFromJSON(const nlohmann::json& in_json);

        /**
         * Globally save currently selected graph in configurator.
         */
        inline bool ConsumeTriggeredGlobalProjectSave(void) {
            bool trigger_global_graph_save = this->graph_state.global_graph_save;
            this->graph_state.global_graph_save = false;
            return trigger_global_graph_save;
        }

        /**
         * Indicates whether project file drop for configurator is valid.
         */
        inline bool IsProjectFileDropValid(void) {
            return this->project_file_drop_valid;
        }

    private:
        // VARIABLES --------------------------------------------------------------

        GraphCollection graph_collection;

        float module_list_sidebar_width;
        ImGuiID selected_list_module_uid;
        ImGuiID add_project_graph_uid;
        ImGuiID module_list_popup_hovered_group_uid;
        bool show_module_list_sidebar;
        bool show_module_list_popup;
        ImVec2 module_list_popup_pos;
        ImGuiID last_selected_callslot_uid;
        megamol::gui::GraphState_t graph_state;
        bool open_popup_load;
        bool project_file_drop_valid;

        // Widgets
        FileBrowserWidget file_browser;
        StringSearchWidget search_widget;
        SplitterWidget splitter_widget;
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        void draw_window_menu(void);
        void draw_window_module_list(float width, float height, bool apply_focus);

        void drawPopUps(void);

        bool load_graph_state_from_file(const std::string& filename);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED

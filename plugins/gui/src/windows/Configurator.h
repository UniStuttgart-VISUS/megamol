/*
 * Configurator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
#define MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED
#pragma once


#include "WindowConfiguration.h"
#include "graph/GraphCollection.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/SplitterWidget.h"
#include "widgets/StringSearchWidget.h"


namespace megamol {
namespace gui {


    class Configurator : public WindowConfiguration {
    public:

        Configurator();
        ~Configurator();

        void Update() override;

        void Draw() override;

        void PopUps() override;

        /**
         * Get hotkey of configurator.
         *
         * @return Hotkeys of configurator.
         */
        inline megamol::gui::HotkeyArray_t& GetHotkeys() {
            return this->graph_state.hotkeys;
        }

        /**
         * Returns required font scalings for graph canvas
         */
        inline const FontScalingArray_t& GetGraphFontScalings() const {
            return this->graph_state.graph_zoom_font_scalings;
        }

        /**
         * Return graph collection.
         */
        GraphCollection& GetGraphCollection() {
            return this->graph_collection;
        }

        /**
         * State to and from JSON.
         */
        bool StateToJSON(nlohmann::json& inout_json);
        bool StateFromJSON(const nlohmann::json& in_json);

        bool SpecificStateFromJSON(const nlohmann::json& in_json) override;
        bool SpecificStateToJSON(nlohmann::json& inout_json) override;

        /**
         * Globally save currently selected graph in configurator.
         */
        inline bool ConsumeTriggeredGlobalProjectSave() {
            bool trigger_global_graph_save = this->graph_state.global_graph_save;
            this->graph_state.global_graph_save = false;
            return trigger_global_graph_save;
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

        // Widgets
        FileBrowserWidget file_browser;
        StringSearchWidget search_widget;
        SplitterWidget splitter_widget;
        HoverToolTip tooltip;

        // FUNCTIONS --------------------------------------------------------------

        void draw_window_menu();
        void draw_window_module_list(float width, float height, bool apply_focus);

        void drawPopUps();

        bool load_graph_state_from_file(const std::string& filename);
    };


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_CONFIGURATOR_H_INCLUDED

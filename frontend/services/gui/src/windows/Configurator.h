/*
 * Configurator.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractWindow.h"
#include "graph/GraphCollection.h"
#include "widgets/FileBrowserWidget.h"
#include "widgets/HoverToolTip.h"
#include "widgets/SplitterWidget.h"
#include "widgets/StringSearchWidget.h"

#include "AbstractWindow2.h"


namespace megamol::gui {


/* ************************************************************************
 * The graph configurator GUI window
 */
class Configurator : public AbstractWindow2 {
public:
    static WindowType GetTypeInfo() {
        WindowType wt;
        wt.unique = true;
        wt.name = "Configurator";
        wt.hotkey = megamol::core::view::KeyCode(megamol::core::view::Key::KEY_F11, core::view::Modifier::NONE);
        return wt;
    }

    explicit Configurator(const std::string& window_name, std::shared_ptr<TransferFunctionEditor> win_tfe_ptr);
    ~Configurator() = default;

    bool Update() override;
    bool Draw() override;
    void PopUps() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

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
     * Globally save currently selected graph in configurator.
     */
    inline bool ConsumeTriggeredGlobalProjectSave() {
        bool trigger_global_graph_save = this->graph_state.global_graph_save;
        this->graph_state.global_graph_save = false;
        return trigger_global_graph_save;
    }

private:
    // VARIABLES --------------------------------------------------------------

    megamol::gui::GraphState_t graph_state;
    GraphCollection graph_collection;

    /** Shortcut pointer to transfer function window */
    std::shared_ptr<TransferFunctionEditor> win_tfeditor_ptr;

    float module_list_sidebar_width;
    ImGuiID selected_list_module_id;
    ImGuiID add_project_graph_uid;
    ImGuiID module_list_popup_hovered_group_uid;
    bool show_module_list_sidebar;
    bool show_module_list_popup;
    ImVec2 module_list_popup_pos;
    ImGuiID last_selected_callslot_uid;
    bool open_popup_load;

    // Widgets
    FileBrowserWidget file_browser;
    StringSearchWidget search_widget;
    SplitterWidget splitter_widget;
    HoverToolTip tooltip;

    // FUNCTIONS --------------------------------------------------------------

    void draw_window_menu();

    void draw_window_module_list(float width, float height, bool omit_focus);
};


} // namespace megamol::gui

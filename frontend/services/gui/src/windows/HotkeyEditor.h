/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once


#include "AbstractWindow.h"
#include "CommandRegistry.h"
#include "WindowCollection.h"
#include "mmcore/MegaMolGraph.h"
#include "widgets/HoverToolTip.h"
#include "widgets/StringSearchWidget.h"


namespace megamol::gui {

class HotkeyEditor : public AbstractWindow {
public:
    explicit HotkeyEditor(const std::string& window_name);
    ~HotkeyEditor();

    bool Update() override;
    bool Draw() override;

    void SpecificStateFromJSON(const nlohmann::json& in_json) override;
    void SpecificStateToJSON(nlohmann::json& inout_json) override;

    void RegisterHotkeys(megamol::core::view::CommandRegistry* cmdregistry, megamol::core::MegaMolGraph* megamolgraph,
        megamol::gui::WindowCollection* wincollection, megamol::gui::HotkeyMap_t* guihotkeys);

private:
    // VARIABLES --------------------------------------------------------------

    int pending_hotkey_assignment;
    megamol::frontend_resources::KeyCode pending_hotkey;

    StringSearchWidget search_widget;
    HoverToolTip tooltip_widget;

    bool is_any_key_down();
    bool is_key_modifier(ImGuiKey k);

    megamol::frontend_resources::CommandRegistry* command_registry_ptr;
    megamol::gui::WindowCollection* window_collection_ptr;
    megamol::gui::HotkeyMap_t* gui_hotkey_ptr;
    megamol::core::MegaMolGraph* megamolgraph_ptr;
    frontend_resources::Command::EffectFunction graph_parameter_lambda;
    frontend_resources::Command::EffectFunction parent_gui_hotkey_lambda;
    frontend_resources::Command::EffectFunction parent_gui_window_lambda;
    frontend_resources::Command::EffectFunction parent_gui_window_hotkey_lambda;
};


} // namespace megamol::gui

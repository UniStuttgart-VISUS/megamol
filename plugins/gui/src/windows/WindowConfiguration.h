/*
 * WindowCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WINDOWCONFIGURATION_INCLUDED
#define MEGAMOL_GUI_WINDOWCONFIGURATION_INCLUDED
#pragma once


#include <functional>
#include <map>
#include <string>
#include <vector>
#include "imgui.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"
#include "mmcore/utility/JSONHelper.h"


namespace megamol {
namespace gui {

    /**
     * This class holds the configuration of a GUI window.
     */
    class WindowConfiguration {
    public:

        /** Identifiers for the predefined window draw callbacks. */
        /// XXX Keep explicit numbers for backward compatibility
        enum WindowConfigID {
            WINDOW_ID_VOLATILE = 0,
            WINDOW_ID_MAIN_PARAMETERS = 1,
            WINDOW_ID_PARAMETERS = 2,
            WINDOW_ID_PERFORMANCE = 3,
            WINDOW_ID_TRANSFER_FUNCTION = 5,
            WINDOW_ID_CONFIGURATOR = 6,
            WINDOW_ID_LOGCONSOLE = 7
        };

        struct BasicConfig {
            bool show = false;                                  // [SAVED] show/hide window
            ImGuiWindowFlags flags = 0;                         // [SAVED] imgui window flags
            core::view::KeyCode hotkey;                         // [SAVED] hotkey for opening/closing window
            ImVec2 position = ImVec2(0.0f, 0.0f);        // [SAVED] position for reset on state loading (current position)
            ImVec2 size = ImVec2(0.0f, 0.0f);            // [SAVED] size for reset on state loading (current size)
            ImVec2 reset_size = ImVec2(0.0f, 0.0f);      // [SAVED] minimum window size for soft reset
            ImVec2 reset_position = ImVec2(0.0f, 0.0f);  // [SAVED] window position for minimize reset
            bool collapsed = false;                             // [SAVED] flag indicating whether window is collapsed or not
            bool reset_pos_size = true;                         // flag indicates whether to reset window position and size
        };

        WindowConfiguration(const std::string& name, WindowConfigID window_id)
                : hash_id(std::hash<std::string>()(name))
                , name(name)
                , window_id(window_id)
                , config() {}

        ~WindowConfiguration() = default;

        std::string Name() const {
            return this->name;
        }

        size_t Hash() const {
            return this->hash_id;
        }

        inline WindowConfigID WindowID() const {
            return this->window_id;
        }

        inline BasicConfig& Config() {
            return this->config;
        }

        std::string FullWindowTitle() const {
            return (this->Name() + "     " + this->config.hotkey.ToString());
        }

        void ApplyWindowSizePosition(bool consider_menu);

        void WindowContextMenu(bool menu_visible, bool& out_collapsing_changed);

        bool StateFromJSON(const nlohmann::json& in_json);

        bool StateToJSON(nlohmann::json& inout_json);

        // --------------------------------------------------------------------
        // IMPLEMENT

        virtual bool SpecificStateToJSON(nlohmann::json& inout_json) { return true; };

        virtual bool SpecificStateFromJSON(const nlohmann::json& in_json) { return true; };

        virtual void Update() { };

        virtual void Draw() { };

        virtual void PopUps() { };

    protected:

        BasicConfig config;

    private:

        size_t hash_id;         // unique hash generated from name to omit string comparison
        std::string name;       // [SAVED] unique name of the window
        WindowConfigID window_id;     // [SAVED] ID of the predefined callback drawing the window content
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWCONFIGURATION_INCLUDED

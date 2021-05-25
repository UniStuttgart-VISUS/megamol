/*
 * WindowCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED
#define MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED
#pragma once


#include <functional>
#include <map>
#include <string>
#include <vector>
#include "imgui.h"
#include "mmcore/utility/JSONHelper.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"


namespace megamol {
namespace gui {

    /**
     * This class holds the configuration of a GUI window.
     */
    class WindowConfiguration {

    public:
        /** Identifiers for the predefined window draw callbacks. */
        /// XXX Keep explicit numbers for backward compatibility
        enum PredefinedCallbackID {
            DRAWCALLBACK_VOLATILE = 0,
            DRAWCALLBACK_MAIN_PARAMETERS = 1,
            DRAWCALLBACK_PARAMETERS = 2,
            DRAWCALLBACK_PERFORMANCE = 3,
            DRAWCALLBACK_TRANSFER_FUNCTION = 5,
            DRAWCALLBACK_CONFIGURATOR = 6,
            DRAWCALLBACK_LOGCONSOLE = 7
        };

        /** Timing mode for performance windows. */
        enum TimingMode { TIMINGMODE_FPS, TIMINGMODE_MS };

        struct Basic {
            bool show = false;          // show/hide window
            ImGuiWindowFlags flags = 0; // imgui window flags
            core::view::KeyCode hotkey; // hotkey for opening/closing window
            ImVec2 position;            // position for reset on state loading (current position)
            ImVec2 size;                // size for reset on state loading (current size)
            ImVec2 reset_size;          // minimum window size for soft reset
            ImVec2 reset_position;      // window position for minimize reset
            bool collapsed = false;     // flag indicating whether window is collapsed or not.
            bool reset_pos_size = true; // [NOT SAVED] set window position and size to fit current viewport
        };

        struct Specific {
            // ---------- Parameter specific configuration ----------
            bool param_show_hotkeys = false;             // flag to toggle showing only parameter hotkeys
            std::vector<std::string> param_modules_list; // modules to show in a parameter window (show all if empty)
            bool param_extended_mode = false;            // flag toggling between Expert and Basic parameter mode.
            // ---------- FPS/MS specific configuration ----------
            bool fpsms_show_options = false;        // show/hide fps/ms options.
            int fpsms_buffer_size = 20;             // maximum count of values in value array
            float fpsms_refresh_rate = 2.0f;        // maximum delay when fps/ms value should be renewed.
            TimingMode fpsms_mode = TIMINGMODE_FPS; // mode for displaying either FPS or MS
            float tmp_current_delay = 0.0f;         // [NOT SAVED] current delay between frames
            std::vector<float> tmp_ms_values;       // [NOT SAVED] current ms values
            std::vector<float> tmp_fps_values;      // [NOT SAVED] current fps values
            float tmp_ms_max = 1.0f;                // [NOT SAVED] current ms plot scaling factor
            float tmp_fps_max = 1.0f;               // [NOT SAVED] current fps plot scaling factor
            // ---------- Transfer Function Editor specific configuration ---------
            bool tfe_view_minimized = false; // flag indicating minimized window state
            bool tfe_view_vertical = false;  // flag indicating vertical window state
            std::string tfe_active_param;    // last active parameter connected to editor
            bool tmp_tfe_reset = false;      // [NOT SAVED] flag for reset of tfe window on state loading
            // ---------- LOG specific configuration ----------
            unsigned int log_level =
                static_cast<int>(megamol::core::utility::log::Log::LEVEL_ALL); // Log level used in log window
            bool log_force_open = true; // flag indicating if log window should be forced open on warnings and errors
        };

        struct Complete {
            Basic basic;
            Specific specific;
        };

        /** Type for predefined window callback function. */
        typedef std::function<void(WindowConfiguration&)> PredefinedCallbackFunc_t;

        /** Type for unknown window callback function. */
        typedef std::function<void(Basic&)> CallbackFunc_t;

        WindowConfiguration(const std::string& name, PredefinedCallbackID cb_id)
                : hash_id(std::hash<std::string>()(name))
                , name(name)
                , callback_id(cb_id)
                , volatile_callback()
                , config() {}

        WindowConfiguration(const std::string& name, CallbackFunc_t& cbf)
                : hash_id(std::hash<std::string>()(name))
                , name(name)
                , callback_id(DRAWCALLBACK_VOLATILE)
                , volatile_callback(cbf)
                , config() {}

        ~WindowConfiguration(void) = default;

        void SetVolatileCallback(CallbackFunc_t& callback) {
            this->volatile_callback = callback;
            this->callback_id = DRAWCALLBACK_VOLATILE;
        }

        std::string Name(void) const {
            return this->name;
        }
        size_t Hash(void) const {
            return this->hash_id;
        }
        PredefinedCallbackID CallbackID(void) const {
            return this->callback_id;
        }
        CallbackFunc_t VolatileCallback(void) const {
            return this->volatile_callback;
        }

        void ApplyWindowSizePosition(bool consider_menu);

        Complete config;

    private:
        size_t hash_id;                   // unique hash generated from name to omit string comparison
        std::string name;                 // unique name of the window
        PredefinedCallbackID callback_id; // ID of the predefined callback drawing the window content
        CallbackFunc_t volatile_callback; // [NOT SAVED] Alternative unknown window callback.
    };

    // --------------------------------------------------------------------

    /**
     * This class controls the placement and appearance of windows.
     */
    class WindowCollection {

    public:
        WindowCollection(void) {
            this->callbacks[WindowConfiguration::PredefinedCallbackID::DRAWCALLBACK_VOLATILE] = nullptr;
        }

        ~WindowCollection(void) = default;

        /**
         * Register callback function for given callback id.
         *
         * @param cbid  The window callback id.
         * @param id    The callback function that should be matched to callback id.
         */
        inline bool RegisterDrawWindowCallback(
            WindowConfiguration::PredefinedCallbackID cb_id, WindowConfiguration::PredefinedCallbackFunc_t cb) {
            if (cb_id == WindowConfiguration::PredefinedCallbackID::DRAWCALLBACK_VOLATILE) {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] DRAWCALLBACK_VOLATILE can not be used for predefined callback. [%s, %s, line %d]\n",
                    __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
            this->callbacks[cb_id] = cb;
            return true;
        }

        /**
         * Draw window content by calling registered callback function in window configuration.
         *
         * @param cbid  The predefined window callback ID.
         */
        inline WindowConfiguration::PredefinedCallbackFunc_t PredefinedWindowCallback(
            WindowConfiguration::PredefinedCallbackID cb_id) {
            return this->callbacks[cb_id]; //
        }

        // --------------------------------------------------------------------
        // CONFIGURATIONs

        /**
         * Add new window.
         *
         * @param window_name    The window name.
         * @param wc  The window configuration.
         */
        bool AddWindowConfiguration(WindowConfiguration& wc);

        /**
         * Enumerate windows and call given function.
         *
         * @param cb  The function to call for enumerated windows.
         */
        inline void EnumWindows(std::function<void(WindowConfiguration&)> cb) {
            // Needs fixed size if window is added while looping
            auto window_count = this->windows.size();
            for (size_t i = 0; i < window_count; i++) {
                cb(this->windows[i]);
            }
        }

        /**
         * Delete window.
         *
         * @param hash_id  The window hash id.
         */
        bool DeleteWindowConfiguration(size_t hash_id);

        bool DeleteWindowConfigurations(void) {
            this->windows.clear();
            return true;
        };

        /**
         * Check if a window configuration for the given window name exists.
         *
         * @param window_name  The window name.
         *
         * @return True if there is a window configuration for the given name, false otherwise.
         */
        inline bool WindowConfigurationExists(size_t hash_id) const {
            for (auto& wc : this->windows) {
                if (wc.Hash() == hash_id)
                    return true;
            }
            return false;
        }

        // --------------------------------------------------------------------
        // STATE

        /**
         * Deserializes a window configuration state.
         * Should be called before(!) ImGui::Begin() because existing window configurations are overwritten.
         *
         * @param json  The JSON to deserialize from.
         *
         * @return True on success, false otherwise.
         */
        bool StateFromJSON(const nlohmann::json& in_json);

        /**
         * Serializes the current window configurations.
         *
         * @param json  The JSON to serialize to.
         *
         * @return True on success, false otherwise.
         */
        bool StateToJSON(nlohmann::json& inout_json);

    private:
        // VARIABLES ------------------------------------------------------

        /** The list of the window names and their configurations. */
        std::map<WindowConfiguration::PredefinedCallbackID, WindowConfiguration::PredefinedCallbackFunc_t> callbacks;

        /** The list of the window configurations. */
        std::vector<WindowConfiguration> windows;
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED

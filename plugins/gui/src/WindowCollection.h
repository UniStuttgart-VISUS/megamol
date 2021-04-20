/*
 * WindowCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED
#define MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED


#include "GUIUtils.h"


namespace megamol {
namespace gui {

    /**
     * This class controls the placement and appearance of windows tied to one GUIView.
     */
    class WindowCollection {

    public:
        /** Identifiers for the window draw callbacks. */
        enum DrawCallbacks {
            DRAWCALLBACK_NONE = 0,
            DRAWCALLBACK_MAIN_PARAMETERS = 1,
            DRAWCALLBACK_PARAMETERS = 2,
            DRAWCALLBACK_PERFORMANCE = 3,
            DRAWCALLBACK_TRANSFER_FUNCTION = 5,
            DRAWCALLBACK_CONFIGURATOR = 6,
            DRAWCALLBACK_LOGCONSOLE = 7
        };

        /** Performance mode for fps/ms windows. */
        enum TimingModes { FPS = 0, MS = 1 };

        /** Module filter mode for parameter windows. */
        enum FilterModes { ALL = 0, INSTANCE = 1, VIEW = 2 };

        /** Struct holding a window configuration. */
        struct WindowConfiguration {
            std::string win_name;           // name of the window
            bool win_show;                  // show/hide window
            bool win_store_config;          // flag indicates whether consiguration of window should be stored or not
            ImGuiWindowFlags win_flags;     // imgui window flags
            DrawCallbacks win_callback;     // ID of the callback drawing the window content
            core::view::KeyCode win_hotkey; // hotkey for opening/closing window
            ImVec2 win_position;            // position for reset on state loading (current position)
            ImVec2 win_size;                // size for reset on state loading (current size)
            ImVec2 win_reset_size;          // minimum window size for soft reset
            ImVec2 win_reset_position;      // window position for minimize reset
            bool win_collapsed;             // flag indicating whether window is collapsed or not.
            bool buf_set_pos_size;          // [NOT SAVED] set window position and size to fit current viewport
            // ---------- Parameter specific configuration ----------
            bool param_show_hotkeys;                     // flag to toggle showing only parameter hotkeys
            std::vector<std::string> param_modules_list; // modules to show in a parameter window (show all if empty)
            FilterModes param_module_filter;             // module filter
            bool param_extended_mode;                    // Flag toggling between Expert and Basic parameter mode.
            // ---------- FPS/MS specific configuration ----------
            bool fpsms_show_options;           // show/hide fps/ms options.
            int fpsms_buffer_size;             // maximum count of values in value array
            float fpsms_refresh_rate;          // maximum delay when fps/ms value should be renewed.
            TimingModes fpsms_mode;            // mode for displaying either FPS or MS
            float buf_current_delay;           // [NOT SAVED] current delay between frames
            std::vector<float> buf_ms_values;  // [NOT SAVED] current ms values
            std::vector<float> buf_fps_values; // [NOT SAVED] current fps values
            float buf_ms_max;                  // [NOT SAVED] current ms plot scaling factor
            float buf_fps_max;                 // [NOT SAVED] current fps plot scaling factor
            // ---------- Transfer Function Editor specific configuration ---------
            bool tfe_view_minimized;      // flag indicating minimized window state
            bool tfe_view_vertical;       // flag indicating vertical window state
            std::string tfe_active_param; // last active parameter connected to editor
            bool buf_tfe_reset;           // [NOT SAVED] flag for reset of tfe window on state loading
            // ---------- LOG specific configuration ----------
            unsigned int log_level; // Log level used in log window
            bool log_force_open;    // Flag indicating if log window should be forced open on warnings and errors

            // Ctor for default values
            WindowConfiguration(void)
                    : win_show(false)
                    , win_store_config(true)
                    , win_flags(0)
                    , win_callback(DrawCallbacks::DRAWCALLBACK_NONE)
                    , win_hotkey(megamol::core::view::KeyCode())
                    , win_position(ImVec2(0.0f, 0.0f))
                    , win_size(ImVec2(0.0f, 0.0f))
                    , win_reset_size(ImVec2(0.0f, 0.0f))
                    , win_reset_position(ImVec2(0.0f, 0.0f))
                    , win_collapsed(false)
                    , buf_set_pos_size(true)
                    // Window specific configurations
                    , param_show_hotkeys(false)
                    , param_modules_list()
                    , param_module_filter(FilterModes::ALL)
                    , param_extended_mode(false)
                    , fpsms_show_options(false)
                    , fpsms_buffer_size(20)
                    , fpsms_refresh_rate(2.0f)
                    , fpsms_mode(TimingModes::FPS)
                    , buf_current_delay(0.0f)
                    , buf_ms_values()
                    , buf_fps_values()
                    , buf_ms_max(1.0f)
                    , buf_fps_max(1.0f)
                    , tfe_view_minimized(false)
                    , tfe_view_vertical(false)
                    , tfe_active_param("")
                    , buf_tfe_reset(false)
                    , log_level(static_cast<int>(megamol::core::utility::log::Log::LEVEL_ALL))
                    , log_force_open(true) {}
        };

        /** Type for callback function. */
        typedef std::function<void(WindowConfiguration& window_config)> GuiCallbackFunc;

        // --------------------------------------------------------------------
        // WINDOWs

        WindowCollection() = default;

        ~WindowCollection(void) = default;

        /**
         * Register callback function for given callback id.
         *
         * @param cbid  The callback id.
         * @param id    The callback function that should be matched to callback id.
         */
        inline bool RegisterDrawWindowCallback(DrawCallbacks cbid, GuiCallbackFunc cb) {
            // Overwrites existing entry with same WindowDrawCallback id.
            this->callbacks[cbid] = cb;
            return true;
        }

        /**
         * Draw window content by calling registered callback function in window configuration.
         *
         * @param window_name  The name of the calling window.
         */
        inline GuiCallbackFunc WindowCallback(DrawCallbacks cbid) {
            // Creates new entry if no callback for cbid is registered (default ctor)
            return this->callbacks[cbid];
        }

        /**
         * Set position and size of currently active window to fit into current viewport.
         *
         * @param window_config  The window configuration.
         */
        void SetWindowSizePosition(WindowConfiguration& window_config, bool consider_menu);

        // --------------------------------------------------------------------
        // CONFIGURATIONs

        /**
         * Add new window.
         *
         * @param window_name    The window name.
         * @param window_config  The window configuration.
         */
        bool AddWindowConfiguration(WindowConfiguration& window_config);

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
         * @param window_name  The window name.
         */
        bool DeleteWindowConfiguration(const std::string& window_name);

        bool DeleteWindowConfigurations(void) {
            this->windows.clear();
            return true;
        };

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
        /**
         * Check if a window configuration for the given window name exists.
         *
         * @param window_name  The window name.
         *
         * @return True if there is a window configuration for the given name, false otherwise.
         */
        inline bool windowConfigurationExists(const std::string& window_name) const {
            for (auto& wc : this->windows) {
                if (wc.win_name == window_name)
                    return true;
            }
            return false;
        }

        // VARIABLES ------------------------------------------------------

        /** The list of the window names and their configurations. */
        std::map<DrawCallbacks, GuiCallbackFunc> callbacks;

        /** The list of the window names and their configurations. */
        std::vector<WindowConfiguration> windows;
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED

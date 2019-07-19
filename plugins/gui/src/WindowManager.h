/*
 * WindowManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "mmcore/view/Input.h"

#include "json.hpp"
#include "vislib/sys/Log.h"

#include <imgui.h>

#include <list>
#include <map>
#include <string>


namespace megamol {
namespace gui {

/**
 * This class controls the placement and appearance of windows tied to one GUIView.
 */
class WindowManager {

public:
    /** Identifiers for the window draw callbacks. */
    enum WindowDrawCallback { NONE = 0, MAIN = 1, PARAM = 2, FPSMS = 3, FONT = 4, TF = 5 };

    /** Performance mode for fps/ms windows. */
    enum TimingMode { FPS = 0, MS = 1 };

    /** Module filter mode for parameter windows. */
    enum FilterMode { ALL = 0, INSTANCE = 1, VIEW = 2 };

    /** Struct holding a window configuration. */
    struct WindowConfiguration {
        bool win_show;                   // show/hide window
        ImGuiWindowFlags win_flags;      // imgui window flags
        WindowDrawCallback win_callback; // id of the callback drawing the window content
        core::view::KeyCode win_hotkey;  // hotkey for opening/closing window
        bool win_reset;        // flag for reset window position and size on state loading  (not saved in state)
        ImVec2 win_position;   // position for reset on state loading (current position)
        ImVec2 win_size;       // size for reset on state loading (current size)
        bool win_soft_reset;   // soft reset of window position and size
        ImVec2 win_reset_size; // minimum window size for soft reset
        // ---------- Parameter specific configuration ----------
        bool param_show_hotkeys;                     // flag to toggle showing only parameter hotkeys
        std::vector<std::string> param_modules_list; // modules to show in a parameter window (show all if empty)
        FilterMode param_module_filter;              // module filter
        // ---------- FPS/MS specific configuration ----------
        bool fpsms_show_options;             // Show/hide fps/ms options.
        int fpsms_max_value_count;           // Maximum count of values in value array
        float fpsms_max_delay;               // Maximum delay when fps/ms value should be renewed.
        TimingMode fpsms_mode;               // mode for displaying either FPS or MS
        float fpsms_current_delay;           // current delay between frames (not saved in state)
        std::vector<float> fpsms_fps_values; // current fps values (not saved in state)
        std::vector<float> fpsms_ms_values;  // current ms values (not saved in state)
        float fpsms_fps_value_scale;         // current scaling factor for fps values (not saved in state)
        float fpsms_ms_value_scale;          // current scaling factor for ms values (not saved in state)
        // ---------- Font specific configuration ----------
        bool font_reset;               // flag for reset font on state loading  (not saved in state)
        std::string font_name;         // the currently used font (only already loaded font names will be restored)
        std::string font_new_filename; // temporary storage of new filename (not saved in state)
        float font_new_size;           // temporary storage of new font size (not saved in state)

        // Ctor for default values
        WindowConfiguration(void)
            : win_show(false)
            , win_flags(0)
            , win_callback(WindowDrawCallback::NONE)
            , win_hotkey(megamol::core::view::KeyCode())
            , win_reset(false)
            , win_position(ImVec2(0.0f, 0.0f))
            , win_size(ImVec2(0.0f, 0.0f))
            , win_soft_reset(true)
            , win_reset_size(ImVec2(500.0f, 300.0f))
            // Window specific configurations
            , param_show_hotkeys(false)
            , param_modules_list()
            , param_module_filter(FilterMode::ALL)
            , fpsms_show_options(false)
            , fpsms_max_value_count(20)
            , fpsms_max_delay(2.0f)
            , fpsms_mode(TimingMode::FPS)
            , fpsms_current_delay(0.0f)
            , fpsms_fps_values()
            , fpsms_ms_values()
            , fpsms_fps_value_scale(1.0f)
            , fpsms_ms_value_scale(1.0f)
            , font_name()
            , font_new_filename()
            , font_new_size(13.0f) {}
    };

    /** Type for callback function. */
    typedef std::function<void(const std::string& window_name, WindowConfiguration& window_config)> GuiCallbackFunc;

    // --------------------------------------------------------------------
    // WINDOWs

    WindowManager() = default;

    ~WindowManager(void) = default;

    /**
     * Register callback function for given callback id.
     *
     * @param cbid  The callback id.
     * @param id    The callback function that should be matched to callback id.
     */
    inline bool RegisterDrawWindowCallback(WindowDrawCallback cbid, GuiCallbackFunc cb) {
        /// Overwrites existing entry with same WindowDrawCallback id.
        this->callbacks[cbid] = cb;
        return true;
    }

    /**
     * Draw window content by calling registered callback function in window configuration.
     *
     * @param window_name  The name of the calling window.
     */
    inline GuiCallbackFunc WindowCallback(WindowDrawCallback cbid) {
        // Creates new entry if no callback for cbid is registered (default ctory)
        return this->callbacks[cbid];
    }

    /**
     * Reset position and size of currently active window to fit into current viewport.
     * Should be triggered via the window configuration flag: soft_reset
     * Processes window configuration flag: soft_reset_size
     * Should be called between ImGui::Begin() and ImGui::End().
     *
     * @param window_name    The window name.
     * @param window_config  The window configuration.
     */
    void SoftResetWindowSizePos(const std::string& window_name, WindowConfiguration& window_config);

    /**
     * Reset position and size after new state has been loaded.
     * Should be triggered via the window configuration flag: state_reset
     * Processes window configuration flags: state_position and state_size
     * Should be called between ImGui::Begin() and ImGui::End().
     *
     * @param window_name    The window name.
     * @param window_config  The window configuration.
     */
    void ResetWindowOnStateLoad(const std::string& window_name, WindowConfiguration& window_config);

    // --------------------------------------------------------------------
    // CONFIGURATIONs

    /**
     * Add new window.
     *
     * @param window_name    The window name.
     * @param window_config  The window configuration.
     */
    bool AddWindowConfiguration(const std::string& window_name, WindowConfiguration& window_config);

    /**
     * Enumerate windows and call given function.
     *
     * @param cb  The function to call for enumerated windows.
     */
    inline void EnumWindows(std::function<void(const std::string&, WindowConfiguration&)> cb) {
        for (auto& wc : this->windows) {
            cb(wc.first, wc.second);
        }
    }

    /**
     * Delete window.
     *
     * @param window_name  The window name.
     */
    bool DeleteWindowConfiguration(const std::string& window_name);

    // --------------------------------------------------------------------
    // STATE

    /**
     * Deserializes a window configuration state.
     * Should be called before(!) ImGui::Begin() because existing window configurations are overwritten.
     *
     * @param json  The string to deserialize from.
     *
     * @return True on success, false otherwise.
     */
    bool StateFromJSON(const std::string& json_string);


    /**
     * Serializes the current window configurations.
     *
     * @param json  The string to serialize to.
     *
     * @return True on success, false otherwise.
     */
    bool StateToJSON(std::string& json_string);

private:
    /**
     * Check if a window configuration for the given window name exists.
     *
     * @param window_name  The window name.
     *
     * @return True if there is a window configuration for the given name, false otherwise.
     */
    inline bool windowConfigurationExists(const std::string& window_name) const {
        return (this->windows.find(window_name) != this->windows.end());
    }

    // VARIABLES ------------------------------------------------------

    /** The list of the window names and their configurations. */
    std::map<WindowDrawCallback, GuiCallbackFunc> callbacks;

    /** The list of the window names and their configurations. */
    std::map<std::string, WindowConfiguration> windows;
};

} // namespace gui
} // namespace megamol

/*
 * WindowManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WINDOWMANAGER_INCLUDED
#define MEGAMOL_GUI_WINDOWMANAGER_INCLUDED


#include "mmcore/view/Input.h"

#include "vislib/sys/Log.h"

#include <list>
#include <map>
#include <string>

#include "json.hpp"

#include "GUIUtils.h"


namespace megamol {
namespace gui {

/**
 * This class controls the placement and appearance of windows tied to one GUIView.
 */
class WindowManager {

public:
    /** Identifiers for the window draw callbacks. */
    enum DrawCallbacks {
        NONE = 0,
        MAIN = 1,
        PARAMETERS = 2,
        PERFORMANCE = 3,
        FONT = 4,
        TRANSFER_FUNCTION = 5,
        CONFIGURATOR = 6
    };

    /** Performance mode for fps/ms windows. */
    enum TimingModes { FPS = 0, MS = 1 };

    /** Module filter mode for parameter windows. */
    enum FilterModes { ALL = 0, INSTANCE = 1, VIEW = 2 };

    /** Struct holding a window configuration. */
    struct WindowConfiguration {
        bool win_show;                  // show/hide window
        bool win_store_config;          // flag indicates whether consiguration of window should be stored or not
        ImGuiWindowFlags win_flags;     // imgui window flags
        DrawCallbacks win_callback;     // id of the callback drawing the window content
        core::view::KeyCode win_hotkey; // hotkey for opening/closing window
        ImVec2 win_position;            // position for reset on state loading (current position)
        ImVec2 win_size;                // size for reset on state loading (current size)
        bool win_soft_reset;            // soft reset of window position and size
        ImVec2 win_reset_size;          // minimum window size for soft reset
        bool win_reset; // flag for reset window position and size on state loading  (not saved in state)
        // ---------- Main window configuration ----------
        std::string main_project_file; // project file name
        // ---------- Parameter specific configuration ----------
        bool param_show_hotkeys;                     // flag to toggle showing only parameter hotkeys
        std::vector<std::string> param_modules_list; // modules to show in a parameter window (show all if empty)
        FilterModes param_module_filter;             // module filter
        // ---------- FPS/MS specific configuration ----------
        bool ms_show_options;          // show/hide fps/ms options.
        int ms_max_history_count;      // maximum count of values in value array
        float ms_refresh_rate;         // maximum delay when fps/ms value should be renewed.
        TimingModes ms_mode;           // mode for displaying either FPS or MS
        float buf_current_delay;       // current delay between frames (not saved in state)
        std::vector<float> buf_values; // current ms values (not saved in state)
        float buf_plot_ms_scaling;     // current ms plot scaling factor (not saved in state)
        float buf_plot_fps_scaling;    // current fps plot scaling factor (not saved in state)
        // ---------- Font specific configuration ---------
        std::string font_name;     // font name (only already loaded font names will be restored)
        bool buf_font_reset;       // flag for reset of font on state loading  (not saved in state)
        std::string buf_font_file; // current font file name (not saved in state)
        float buf_font_size;       // current font size (not saved in state)

        // Ctor for default values
        WindowConfiguration(void)
            : win_show(false)
            , win_store_config(true)
            , win_flags(0)
            , win_callback(DrawCallbacks::NONE)
            , win_hotkey(megamol::core::view::KeyCode())
            , win_position(ImVec2(0.0f, 0.0f))
            , win_size(ImVec2(0.0f, 0.0f))
            , win_soft_reset(true)
            , win_reset_size(ImVec2(0.0f, 0.0f))
            , win_reset(true)
            // Window specific configurations
            , main_project_file("")
            , param_show_hotkeys(false)
            , param_modules_list()
            , param_module_filter(FilterModes::ALL)
            , ms_show_options(false)
            , ms_max_history_count(20)
            , ms_refresh_rate(2.0f)
            , ms_mode(TimingModes::FPS)
            , buf_current_delay(0.0f)
            , buf_values()
            , buf_plot_ms_scaling(1.0f)
            , buf_plot_fps_scaling(1.0f)
            , font_name()
            , buf_font_reset(false)
            , buf_font_file()
            , buf_font_size(13.0f) {}
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
    inline bool RegisterDrawWindowCallback(DrawCallbacks cbid, GuiCallbackFunc cb) {
        /// Overwrites existing entry with same WindowDrawCallback id.
        this->callbacks[cbid] = cb;
        return true;
    }

    /**
     * Draw window content by calling registered callback function in window configuration.
     *
     * @param window_name  The name of the calling window.
     */
    inline GuiCallbackFunc WindowCallback(DrawCallbacks cbid) {
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

    /** Utils being used all over the place */
    GUIUtils utils;

    /** The list of the window names and their configurations. */
    std::map<DrawCallbacks, GuiCallbackFunc> callbacks;

    /** The list of the window names and their configurations. */
    std::map<std::string, WindowConfiguration> windows;
};

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWMANAGER_INCLUDED

/*
 * GUIWindowManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#ifndef MEGAMOL_GUI_WINDOWMANAGER_H_INCLUDED
#define MEGAMOL_GUI_WINDOWMANAGER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#    pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */
#if defined(_WIN32) && defined(_MANAGED)
#    pragma managed(push, off)
#endif /* defined(_WIN32) && defined(_MANAGED) */


#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <string>

#include <imgui.h>

#include "json.hpp"

#include "mmcore/view/Input.h"
#include "vislib/sys/Log.h"

#include "GUIUtility.h"


namespace megamol {
namespace gui {

/**
 * Managing window configurations for GUI
 *
 * Profile changes are read/written directly from/to the profile file.
 *
 */
class GUIWindowManager : public GUIUtility {

public:
    /**
     * Ctor
     */
    GUIWindowManager(std::string filename = "mmgui.profile");

    /**
     * Dtor
     */
    ~GUIWindowManager(void);

    /** Identifiers for the window draw callbacks. */
    enum WindowDrawCallback { NONE = 0, MAIN = 1, PARAM = 2, FPSMS = 3, FONT = 4, TF = 5 };

    /** Performance mode for fps/ms windows. */
    enum FpsMsMode { FPS = 0, MS = 1 };

    /** Module filter mode for parameter windows. */
    enum FilterMode { ALL = 0, INSTANCE = 1, VIEW = 2 };

    /** Struct holding a window configuration. */
    struct WindowConfiguration {
        bool win_show;                   // show/hide window
        ImGuiWindowFlags win_flags;      // imgui window flags
        WindowDrawCallback win_callback; // id of the callback drawing the window content
        core::view::KeyCode win_hotkey;  // hotkey for opening/closing window
        bool win_reset;        // flag for reset window position and size on profile loading  (not saved in profile)
        ImVec2 win_position;   // position for reset on profile loading (current position)
        ImVec2 win_size;       // size for reset on profile loading (current size)
        bool win_soft_reset;   // soft reset of window position and size
        ImVec2 win_reset_size; // minimum window size for soft reset
        // ---------- Parameter specific condfiguration ----------
        bool param_show_hotkeys;                     // flag to toggle showing only parameter hotkeys
        std::vector<std::string> param_modules_list; // modules to show in a parameter window (show all if empty)
        FilterMode param_module_filter;              // module filter
        // ---------- FPS/MS specific condfiguration ----------
        bool fpsms_show_options;             // Show/hide fps/ms options.
        int fpsms_max_value_count;           // Maximum count of values in value array
        float fpsms_max_delay;               // Maximum delay when fps/ms value should be renewed.
        FpsMsMode fpsms_mode;                // mode for displaying either FPS or MS
        float fpsms_current_delay;           // current delay between frames (not saved in profile)
        std::vector<float> fpsms_fps_values; // current fps values (not saved in profile)
        std::vector<float> fpsms_ms_values;  // current ms values (not saved in profile)
        float fpsms_fps_value_scale;         // current scaling factor for fps values (not saved in profile)
        float fpsms_ms_value_scale;          // current scaling factor for ms values (not saved in profile)
        // ---------- Font specific condfiguration ----------
        bool font_reset;               // flag for reset font on profile loading  (not saved in profile)
        std::string font_name;         // the currently used font (only already loaded font names will be restored)
        std::string font_new_filename; // temporary storage of new filename (not saved in profile)
        float font_new_size;           // temporary storage of new font size (not saved in profile)

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
            , fpsms_mode(FpsMsMode::FPS)
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
     * Reset position and size after new profile has been loaded.
     * Should be triggered via the window configuration flag: profile_reset
     * Processes window configuration flags: profile_position and profile_size
     * Should be called between ImGui::Begin() and ImGui::End().
     *
     * @param window_name    The window name.
     * @param window_config  The window configuration.
     */
    void ResetWindowOnProfileLoad(const std::string& window_name, WindowConfiguration& window_config);

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
    // PROFILEs

    /**
     * Returns a list of the currently available window configuration profiles.
     */
    std::list<std::string> GetWindowConfigurationProfileList(void);

    /**
     * Load a window configuration profile.
     * Should be called before(!) ImGui::Begin() because existing window configurations are overwritten.
     *
     * @param profile_name  The profile name.
     *
     * @return True on success, false otherwise.
     */
    bool LoadWindowConfigurationProfile(const std::string& profile_name);

    /**
     * Delete a window configuration profile.
     *
     * @param profile_name  The profile name.
     *
     * @return True on success, false otherwise.
     */
    bool DeleteWindowConfigurationProfile(const std::string& profile_name);

    /**
     * Save the current window configurations to a new profile.
     *
     * @param profile_name  The profile name.
     *
     * @return True on success, false otherwise.
     */
    bool SaveWindowConfigurationProfile(const std::string& profile_name);

private:
    /**
     * Saving current window configurations to file.
     *
     * @return True on success, false otherwise.
     */
    bool saveWindowConfigurationFile(nlohmann::json& in_profiles);

    /**
     * Loading window configurations from file.
     *
     * @return True on success, false otherwise.
     */
    bool loadWindowConfigurationFile(nlohmann::json& out_profiles);

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

    /** The file the window configuration profiles are stored to. */
    std::string filename;
};


} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWMANAGER_H_INCLUDED
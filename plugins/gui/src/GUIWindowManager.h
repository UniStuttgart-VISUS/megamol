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


#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include <imgui.h>

#include "json.hpp"

#include "mmcore/view/Input.h"
#include "vislib/sys/Log.h"

#include "GUIUtility.h"


namespace megamol {
namespace gui {

/**
 * Managing window configurations for GUI
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
    enum WindowDrawCallback {
        NONE   = 0,
        MAIN   = 1,
        PARAM  = 2,
        FPSMS  = 3,
        FONT   = 4,
        TF     = 5
    };

    /** Performance mode for fps/ms windows. */
    enum FpsMsMode { FPS = 0, MS = 1 };

    /** Module filter mode for parameter windows. */
    enum FilterMode { ALL = 0, INSTANCE = 1, VIEW = 2 };

    /** Additional configuration varaibles for parameter windows. */
    struct ParamConfig {
        bool                     show_hotkeys;  // flag to toggle showing only parameter hotkeys
        std::vector<std::string> modules_list;  // modules to show in a parameter window (show all if empty)
        FilterMode               module_filter; // module filter

        // Ctor for default values
        ParamConfig(void) : 
            show_hotkeys(false)
            , modules_list()
            , module_filter(FilterMode::ALL) {
        }
    };

    /** Additional configuration varaibles for fps/ms windows. */
    struct FpsMsConfig {
        bool                show_options;           // Show/hide fps/ms options. 
        size_t              max_value_count;        // Maximum count of values in value array
        float               max_delay;              // Maximum delay when fps/ms value should be renewed.
        FpsMsMode           mode;                   // mode for displaying either FPS or MS

        float               current_delay;          // current delay between frames (not saved in profile)
        std::vector<float>  fps_values;             // current fps values (not saved in profile)
        std::vector<float>  ms_values;              // current ms values (not saved in profile)

        // Ctor for default values
        FpsMsConfig(void) :
            show_options(false)
            , max_value_count(20)
            , max_delay(2.0f)
            , mode(FpsMsMode::FPS)
            , current_delay(0.0f)
            , fps_values()
            , ms_values() {
        }
    };

    /** Additional configuration varaibles for font selection windows. */
    struct FontConfig {
        std::string font_name;                  // the currently used font (only already loaded font names can be restored on profile change)

        // Ctor for default values
        FontConfig(void) :
            font_name() {
        }
    };

    /** Additional configuration varaibles for transfer function windows. */
    struct TfConfig {
        // Ctor for default values
        TfConfig(void) {
        }
    };

    /** Struct holding a window configuration. */
    struct WindowConfiguration {
        bool                show;               // show/hide window
        ImGuiWindowFlags    flags;              // imgui window flags
        WindowDrawCallback  callback;           // id of the callback drawing the window content
        core::view::KeyCode hotkey;             // hotkey for opening/closing window
        bool                reset;              // flag for reset window position and size on profile loading
        ImVec2              position;           // position for reset on profile loading (current position)
        ImVec2              size;               // size for reset on profile loading (current size)
        bool                soft_reset;         // soft reset of window position and size
        ImVec2              reset_size;         // minimum window size for soft reset
        // ---------- Window specific condfigurations ----------
        ParamConfig         param_config;
        FpsMsConfig         fpsms_config;
        FontConfig          font_config;
        TfConfig            tf_config;

        // Ctor for default values
        WindowConfiguration(void) :
            show(false)
            , flags(0)
            , callback(WindowDrawCallback::NONE)
            , hotkey(megamol::core::view::KeyCode())
            , reset(false)
            , position(ImVec2(0.0f, 0.0f))
            , size(ImVec2(0.0f, 0.0f))
            , soft_reset(true)
            , reset_size(ImVec2(500.0f, 300.0f))
            // Window specific condfigurations
            , param_config()
            , fpsms_config()
            , font_config()
            , tf_config() {
        }
    } ;

    /** Type for callback function. */
    typedef std::function<void(const std::string& window_name, WindowConfiguration& window_config)> GuiCallbackFunc;

    // --------------------------------------------------------------------
    //WINDOWs

    /**
     * Register callback function for given callback id.
     *
     * @param cbid  The callback id.
     * @param id    The callback function that should be matched to callback id.
     */
    bool RegisterDrawWindowCallback(WindowDrawCallback cbid, GuiCallbackFunc cb);

    /**
     * Draw window content by calling registered callback function in window configuration.
     *
     * @param window_name  The name of the calling window.
     */
    inline bool DrawWindowContent(const std::string& window_name);

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
    //CONFIGURATIONs

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
    void EnumWindows(std::function<void(const std::string&, WindowConfiguration&)> cb);

    /**
     * Delete window.
     *
     * @param window_name  The window name.
     */
     //bool DeleteWindowConfiguration(const std::string& window_name);

    // --------------------------------------------------------------------
    //PROFILEs

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
    bool saveWindowConfigurationFile(void);

    /**
     * Loading window configurations from file.
     *
     * @return True on success, false otherwise.
     */
    bool loadWindowConfigurationFile(void);

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

    /** The the current window configuration profiles as JSON. */
    nlohmann::json profiles;

};
 

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWMANAGER_H_INCLUDED
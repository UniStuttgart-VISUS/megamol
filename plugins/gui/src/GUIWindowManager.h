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

    /** Type for holding a window configuration. */
    typedef struct _win_config {
        bool                         show;                // show/hide window
        ImGuiWindowFlags             flags;               // imgui window flags
        int                          draw_func_id;        // id of the function drawing the window content
        megamol::core::view::KeyCode hotkey;              // hotkey for opening/closing window
        bool                         profile_reset;       // reset window position and size on profile loading
        ImVec2                       profile_position;    // keeping last position for reset on profile loading
        ImVec2                       profile_size;        // keeping last size for reset on profile loading
        bool                         soft_reset;          // reset window position and size
        ImVec2                       soft_reset_size;     // default width and height of window (ignored when auto resize flag is setS)
        bool                         show_hotkeys;        // flag to toggle showing only parameter hotkeys
        std::vector<std::string>     param_modules;       // modules to show in a parameter window
    } WindowConfiguration;

    /**
     * Ctor
     */
    GUIWindowManager(std::string filename = "mmgui.profile");

    /**
     * Dtor
     */
    ~GUIWindowManager(void);

    /**
     * Get pointer to specific window configuration.
     *
     * @param window_name  The window name.
     */
    WindowConfiguration* GetWindowConfiguration(std::string window_name) {
        if (!this->windowConfigurationExists(window_name)) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[GUIWindowManager][GetWindowConfiguration] Didn't find existing window '%s'.", window_name.c_str());
            return nullptr;
        }
        return &this->windows[window_name];
    }

    /**
     * Add new window.
     *
     * @param The window name.
     */
    bool AddWindowConfiguration(std::string window_name, WindowConfiguration& config);

    /**
     * Delete window.
     *
     * @param window_name  The window name.
     */
    //bool DeleteWindowConfiguration(std::string window_name);

    /**
     * Enumerate windows and call given function.
     *
     * @param cb  The function to call for enumerated windows.
      */
    void EnumWindows(std::function<void(const std::string&, WindowConfiguration&)> cb);

    /**
     * Reset position and size of currently active window to fit into current viewport.
     * (Should be called between ImGui::Begin() and ImGui::End())
     *
     * @param window_name  The window name.
     */
    void SoftResetWindowSizePos(std::string window_name);

    // --------------------------------------------------------------------

    /**
     * Returns a list of the currently available window configuration profiles.
     */
    std::list<std::string> GetWindowConfigurationProfileList(void);

    /**
     * Load a window configuration profile.
     *
     * @param profile_name  The profile name.
     *
     * @return True on success, false otherwise.
     */
    bool LoadWindowConfigurationProfile(std::string profile_name);

    /**
     * Delete a window configuration profile.
     *
     * @param profile_name  The profile name.
     *
     * @return True on success, false otherwise.
     */
    bool DeleteWindowConfigurationProfile(std::string profile_name);

    /**
     * Save the current window configurations to a new profile.
     *
     * @param profile_name  The profile name.
     *
     * @return True on success, false otherwise.
     */
    bool SaveWindowConfigurationProfile(std::string profile_name);

    /**
     * Reset position and size after new profile has been loaded.
     * (Should be called between ImGui::Begin() and ImGui::End())
     *
     * @param window_name  The window name.
     */
    void ResetWindowSizePosOnProfileLoad(std::string window_name);

private:

    // VARIABLES ------------------------------------------------------

    /** The list of the window names and their configurations. */
    std::map<std::string, WindowConfiguration> windows;

    /** The file the window configuration profiles are stored to. */
    std::string filename;

    /** The the current window configuration profiles as JSON. */
    nlohmann::json profiles;

    // FUNCTIONS ------------------------------------------------------

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
     * @param name  The window name.
     *
     * @return True if there is a window configuration for the given name, false otherwise.
     */
    inline bool windowConfigurationExists(std::string& name) const {
        return (this->windows.find(name) != this->windows.end());
    }

};
 

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWMANAGER_H_INCLUDED
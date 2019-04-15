/*
 * GUIWindowManager.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#ifndef MEGAMOL_GUI_GUISETTINGS_H_INCLUDED
#define MEGAMOL_GUI_GUISETTINGS_H_INCLUDED
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
        ImVec2                       position;            // 
        ImVec2                       size;                // 
        bool                         show;                // show/hide window
        bool                         reset;               // reset window position and size
        ImVec2                       default_size;        // default width and height of window (ignored when auto resize flag is setS)
        megamol::core::view::KeyCode hotkey;              // hotkey for opening/closing window
        ImGuiWindowFlags             flags;               // imgui window flags
        int                          draw_func_id;        // id of the function drawing the window content
        bool                         param_hotkeys_show;  // flag to toggle showing only parameter hotkeys
        bool                         param_main;          // flag indicating main parameter window
        std::vector<std::string>     param_mods;          // modules to show in a parameter window
    } WindowConfiguration;

    /**
     * Ctor
     */
    GUIWindowManager(std::string inifilename = "mmgui.ini");

    /**
     * Dtor
     */
    ~GUIWindowManager(void);

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
    bool DeleteWindowConfiguration(std::string window_name);

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
     * Enumerate windows and call given function.
     *
     * @param cb  The function to call for enumerated windows.
      */
    void EnumWindows(std::function<void(const std::string&, WindowConfiguration&)> cb);

    /**
     * Reset position and size of currently active window.
     * (Should be called between ImGui::Begin() and ImGui::End())
     *
     * @param window_name  The window name.
     */
    void ResetWindowSizePos(std::string window_name);

    // --------------------------------------------------------------------


    /**
     * ...
     */
    std::list<std::string> GetWindowSettingsProfileList(void);

    /**
     * ...
     */
    bool LoadWindowSettingsProfile(std::string profile_name);

    /**
     * ...
     */
    bool DeleteWindowSettingsProfile(std::string profile_name);

    /**
     * ...
     */
    bool SaveWindowSettingsProfie(std::string profile_name);

    /** 
     * Set file name for writing the window settings.
     */
    //inline void SetIniFilenamename(std::string file) {
    //    this->inifilename = file;
    //}

    /**
     * Get the current file name the window settings are written to.
     */
    //inline std::string GetIniFilenamename(void) {
    //    return this->inifilename;
    //}

private:

    // VARIABLES ------------------------------------------------------

    /** The list of the window names and their configurations. */
    std::map<std::string, WindowConfiguration> windows;

    /** The file the settings */
    std::string inifilename;

    /** The settings in JSON format. */
    nlohmann::json settings_store;

    // FUNCTIONS ------------------------------------------------------

    /**
     * ...
     */
    bool saveWindowSettingsFile(void);

    /**
     * ...
     */
    bool loadWindowSettingsFile(void);

    /**
     * ...
     */
    inline bool windowConfigurationExists(std::string& name) const {
        return (this->windows.find(name) != this->windows.end());
    }

};
 

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUISETTINGS_H_INCLUDED
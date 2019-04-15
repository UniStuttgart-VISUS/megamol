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
        template <typename F> class GUIWindowManager : public GUIUtility {

        public:

            /** Type for holding a window configuration. */
            typedef struct _win_config {
                bool show;                           // show/hide window
                bool reset;                          // reset window position and size
                megamol::core::view::KeyCode hotkey; // hotkey for opening/closing window
                ImGuiWindowFlags flags;              // imgui window flags
                F func;                              // pointer to function drawing window content
                ImVec2 dim;                          // default width and height of window (ignored when auto resize flag is setS)
                bool param_hotkeys_show;             // flag to toggle showing only parameter hotkeys
                bool param_main;                     // flag indicating main parameter window
                std::vector<std::string> param_mods; // modules to show in a parameter window
            } WindowConfiguration;

            /**
             * Ctor
             */
            GUIWindowManager();

            /**
             * Dtor
             */
            ~GUIWindowManager(void);

            /**
             * ...
             */
            bool AddWindowConfiguration(std::string name, WindowConfiguration& config) {
                if (this->windowConfigurationExists(name)) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[GUIWindowManager][AddWindowConfiguration] Found existing window configuration for '%s'. Window name should be unique.", name.c_str());
                }
                this->windows.emplace(name, config);
                return true;
            }

            /**
             * ...
             */
            inline WindowConfiguration* GetWindowConfiguration(std::string name) {
                if (!this->windowConfigurationExists(name)) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[GUIWindowManager][AddWindowConfiguration] Didn't find existing window configuration for '%s'. Window name should exist.", name.c_str());
                    return nullptr;
                }
                return &this->windows[name];
            }

            /**
             * ...
             */
            void EnumWindows(std::function<void(const std::string&, WindowConfiguration&)> cb);

            /**
             * ...
             *
             * Should be called between ImGui::Begin() and ImGui::End()
             */
            void ResetWindowSizePos(std::string win_name);

            /** 
             * Set file name for writing the settings.
             */
            inline void SetIniFilename(std::string file) {
                this->inifile = file;
            }

        private:

            // VARIABLES ------------------------------------------------------

            /** The list of the window names and their configurations. */
            std::map<std::string, WindowConfiguration> windows;

            /** The file the settings */
            std::string inifile;

            /** The settings in JSON format. */
            nlohmann::json settings;

            // FUNCTIONS ------------------------------------------------------

            inline bool windowConfigurationExists(std::string& win_name) const {
                return (this->windows.find(win_name) != this->windows.end());
            }

        };
 

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_GUISETTINGS_H_INCLUDED


/**
 * GUIWindowManager<F>::Ctor
 */
template <typename F> megamol::gui::GUIWindowManager<F>::GUIWindowManager(void) :
    windows()
    , inifile("mmgui.ini")
    , settings() {

    // nothing to do here ...
}


/**
 * GUIWindowManager<F>::Dtor
 */
template <typename F> megamol::gui::GUIWindowManager<F>::~GUIWindowManager(void) {

    this->windows.clear();
}


/**
 * GUIWindowManager<F>::EnumWindows
 */
template <typename F> void megamol::gui::GUIWindowManager<F>::EnumWindows(std::function<void(const std::string&, WindowConfiguration&)> cb) {

    for (auto &wc : this->windows) {
        cb(wc.first, wc.second);
    }
}


/**
 * GUIWindowManager<F>::ResetWindowSizePos
 */
template <typename F> void megamol::gui::GUIWindowManager<F>::ResetWindowSizePos(std::string win_name) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    auto win = this->GetWindowConfiguration(win_name);
    float width = win->dim.x;
    float height = win->dim.y;

    auto win_pos = ImGui::GetWindowPos();
    if (win_pos.x < 0) {
        win_pos.x = style.DisplayWindowPadding.x;
    }
    if (win_pos.y < 0) {
        win_pos.y = style.DisplayWindowPadding.y;
    }

    ImVec2 win_size;
    if (win->flags | ImGuiWindowFlags_AlwaysAutoResize) {
        win_size = ImGui::GetWindowSize();
    }
    else {
        win_size = ImVec2(width, height);
    }
    float win_width = io.DisplaySize.x - (win_pos.x + style.DisplayWindowPadding.x);
    if (win_width < win_size.x) {
        win_pos.x = io.DisplaySize.x - (win_size.x + style.DisplayWindowPadding.x);
    }
    float win_height = io.DisplaySize.y - (win_pos.y + style.DisplayWindowPadding.y);
    if (win_height < win_size.y) {
        win_pos.y = io.DisplaySize.y - (win_size.y + style.DisplayWindowPadding.y);
    }

    ImGui::SetWindowSize(win_name.c_str(), ImVec2(width, height), ImGuiCond_Always);
    ImGui::SetWindowPos(win_name.c_str(), win_pos, ImGuiCond_Always);
}
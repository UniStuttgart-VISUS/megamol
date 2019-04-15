/*
 * GUIWindowManager.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "GUIWindowManager.h"


using namespace megamol::gui;


/**
 * GUIWindowManager::Ctor
 */
GUIWindowManager::GUIWindowManager(std::string inifilename) :
    windows()
    , inifilename(inifilename)
    , settings_store() {

    this->loadWindowSettingsFile();
}


/**
 * GUIWindowManager::Dtor
 */
GUIWindowManager::~GUIWindowManager(void) {

    this->windows.clear();
}


/**
 * GUIWindowManager::AddWindowConfiguration
 */
bool GUIWindowManager::AddWindowConfiguration(std::string window_name, WindowConfiguration& config) {

    if (this->windowConfigurationExists(window_name)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[GUIWindowManager] Found already existing window '%s'. Window name must be unique.", window_name.c_str());
        return false;
    }
    this->windows.emplace(window_name, config);
    return true;
}


/**
 * GUIWindowManager::DeleteWindowConfiguration
 */
bool GUIWindowManager::DeleteWindowConfiguration(std::string window_name) {

    if (!this->windowConfigurationExists(window_name)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[GUIWindowManager] Found no existing window '%s'.", window_name.c_str());
        return false;
    }
    this->windows.erase(window_name);
    return true;
}


/**
 * GUIWindowManager::EnumWindows
 */
void GUIWindowManager::EnumWindows(std::function<void(const std::string&, GUIWindowManager::WindowConfiguration&)> cb) {

    for (auto &wc : this->windows) {
        cb(wc.first, wc.second);
    }
}


/**
 * GUIWindowManager::ResetWindowSizePos
 */
void GUIWindowManager::ResetWindowSizePos(std::string window_name) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    auto win = this->GetWindowConfiguration(window_name);
    float width = win->default_size.x;
    float height = win->default_size.y;

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

    ImGui::SetWindowSize(window_name.c_str(), ImVec2(width, height), ImGuiCond_Always);
    ImGui::SetWindowPos(window_name.c_str(), win_pos, ImGuiCond_Always);
}


/**
 * GUIWindowManager::GetWindowSettingsProfileList
 */
std::list<std::string> GUIWindowManager::GetWindowSettingsProfileList(void) {

    std::list<std::string> out_list;

    for (auto &p : this->settings_store.items()) {
        out_list.emplace_back(p.key());
    }

    return out_list;
}


/**
 * GUIWindowManager::LoadWindowSettingsProfile
 */
bool GUIWindowManager::LoadWindowSettingsProfile(std::string profile_name) {

    std::map<std::string, WindowConfiguration> tmp_windows;
    bool found_main_window = false;

    for (auto &p : this->settings_store.items()) {
        // Search for profile 
        if (p.key() == profile_name) {
            // Loop over all windows
            for (auto &w : p.value().items()) {
                std::string window_name = w.key();
                WindowConfiguration tmp_config;

                // Getting all configuration values
                try
                {

                    //auto config_values = w.value();
                    //if (config_values.at("position").is_array() && (config_values.at("position").size() == 2)) {
                    //    config_values.at("position")[0].get_to(tmp_config.position.x);
                    //    config_values.at("position")[1].get_to(tmp_config.position.y);
                    //}


                        //tmp_config.position            = ImVec2(0.0f, 0.0f);
                        //tmp_config.size                = ImVec2(0.0f, 0.0f);
                        //tmp_config.default_size        = ImVec2(500.0f, 300.0f);
                        //tmp_config.reset               = true;
                        //tmp_config.show                = true;
                        //tmp_config.hotkey              = core::view::KeyCode(core::view::Key::KEY_F12);
                        //tmp_config.flags               = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_HorizontalScrollbar;
                        //tmp_config.draw_func_id        = 0;
                        //tmp_config.param_main          = true;
                        //tmp_config.param_hotkeys_show  = false;
                        //tmp_config.param_mods.clear();
                    

                }
                catch (...) {
                    vislib::sys::Log::DefaultLog.WriteError("[GUIWindowManager] Error reading profile '%s'", profile_name.c_str());
                    return false;
                }
                if (tmp_config.param_main) {
                    found_main_window = true;
                }

                tmp_windows.emplace(window_name, tmp_config);
            }
            if (found_main_window) {
                this->windows = tmp_windows;
                return true;
            }
            else {
                vislib::sys::Log::DefaultLog.WriteError("[GUIWindowManager] Couldn't find main window");
            }
        }
    }

    return false;
}

/**
 * GUIWindowManager::DeleteWindowSettingsProfile
 */
bool GUIWindowManager::DeleteWindowSettingsProfile(std::string profile_name) {

    if (this->settings_store.erase(profile_name) > 0) {
        return this->saveWindowSettingsFile();
    }

    return false;
}

/**
 * GUIWindowManager::SaveWindowSettingsProfie
 */
bool GUIWindowManager::SaveWindowSettingsProfie(std::string profile_name) {

    // (Overwriting existing profile with same name)

    for (auto& w : this->windows) {
        std::string window_name = w.first;
        WindowConfiguration window_config = w.second;
        this->settings_store[profile_name][window_name]["position"]           = { window_config.position.x, window_config.position.y };
        this->settings_store[profile_name][window_name]["size"]               = { window_config.size.x, window_config.size.y };
        this->settings_store[profile_name][window_name]["show"]               = window_config.show;
        this->settings_store[profile_name][window_name]["reset"]              = window_config.reset;
        this->settings_store[profile_name][window_name]["default_size"]       = { window_config.default_size.x, window_config.default_size.y };
        this->settings_store[profile_name][window_name]["hotkey"]             = { (int)(window_config.hotkey.Key()), window_config.hotkey.Modifiers().toInt() };
        this->settings_store[profile_name][window_name]["flags"]              = (int)(window_config.flags);
        this->settings_store[profile_name][window_name]["draw_func_id"]       = window_config.draw_func_id;
        this->settings_store[profile_name][window_name]["param_hotkeys_show"] = window_config.param_hotkeys_show;
        this->settings_store[profile_name][window_name]["param_main"]         = window_config.param_main;
        this->settings_store[profile_name][window_name]["param_mods"]         = window_config.param_mods;
    }

    return this->saveWindowSettingsFile();
}

/**
 * GUIWindowManager::saveWindowSettingsFile
 */
bool GUIWindowManager::saveWindowSettingsFile(void) {

    // (Overwriting existing file with same name)

    std::ofstream inifile;
    inifile.open(this->inifilename);

    if (inifile.is_open() && inifile.good()) {
        inifile << this->settings_store.dump(4);
        inifile.close();
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "[GUIWindowManager] Couldn't write to ini file: '%s'", this->inifilename.c_str());

    return false;
}


/**
 * GUIWindowManager::loadWindowSettingsFile
 */
bool GUIWindowManager::loadWindowSettingsFile(void) {

    std::ifstream inifile;
    inifile.open(this->inifilename);

    std::string line;
    std::stringstream stream;


    if (inifile.is_open() && inifile.good()) {
        while (std::getline(inifile, line))
        {
            stream << line << std::endl;
        }
        inifile.close();

        nlohmann::json parsed_json = nlohmann::json::parse(stream.str());

        // Check for valid JSON object
        if (!parsed_json.is_object()) {
            vislib::sys::Log::DefaultLog.WriteError(
                "[GUIWindowManager] File content is no valid JSON object.");
            return false;
        }

        this->settings_store = parsed_json;

        return true;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "[GUIWindowManager] Couldn't read ini file: '%s'", this->inifilename.c_str());

    return false;
}

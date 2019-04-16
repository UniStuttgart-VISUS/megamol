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
GUIWindowManager::GUIWindowManager(std::string filename) :
    windows()
    , filename(filename)
    , profiles() {

    this->loadWindowConfigurationFile();
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
//bool GUIWindowManager::DeleteWindowConfiguration(std::string window_name) {
//
//    if (!this->windowConfigurationExists(window_name)) {
//        vislib::sys::Log::DefaultLog.WriteError(
//            "[GUIWindowManager] Found no existing window '%s'.", window_name.c_str());
//        return false;
//    }
//    this->windows.erase(window_name);
//    return true;
//}


/**
 * GUIWindowManager::EnumWindows
 */
void GUIWindowManager::EnumWindows(std::function<void(const std::string&, GUIWindowManager::WindowConfiguration&)> cb) {

    for (auto &wc : this->windows) {
        cb(wc.first, wc.second);
    }
}


/**
 * GUIWindowManager::SoftResetWindowSizePos
 */
void GUIWindowManager::SoftResetWindowSizePos(std::string window_name) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    auto win = this->GetWindowConfiguration(window_name);
    float width = win->soft_reset_size.x;
    float height = win->soft_reset_size.y;

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
 * GUIWindowManager::ResetWindowSizePosOnProfileLoad
 */
void GUIWindowManager::ResetWindowSizePosOnProfileLoad(std::string window_name) {

    assert(ImGui::GetCurrentContext() != nullptr);

    WindowConfiguration* win_config = this->GetWindowConfiguration(window_name);
    ImVec2 pos = win_config->profile_position;
    ImVec2 size = win_config->profile_size;

    ImGui::SetWindowSize(window_name.c_str(), size, ImGuiCond_Always);
    ImGui::SetWindowPos(window_name.c_str(), pos, ImGuiCond_Always);
}


/**
 * GUIWindowManager::LoadWindowConfigurationProfile
 */
bool GUIWindowManager::LoadWindowConfigurationProfile(std::string profile_name) {

    bool check = true;
    std::map<std::string, WindowConfiguration> tmp_windows;

    for (auto &p : this->profiles.items()) {
        // Search for profile 
        if (p.key() == profile_name) {
            // Loop over all windows
            for (auto &w : p.value().items()) {
                std::string window_name = w.key();
                WindowConfiguration tmp_config;
                // Getting all configuration values for current window.
                try
                {
                    auto config_values = w.value();

                    // show
                    if (config_values.at("show").is_boolean()) {
                        config_values.at("show").get_to(tmp_config.show);
                    }
                    else {
                        check = false;
                    }
                    // flags
                    if (config_values.at("flags").is_number_integer()) {
                        tmp_config.flags = (ImGuiWindowFlags)config_values.at("flags").get<int>();
                    }
                    else {
                        check = false;
                    }
                    // draw_func_id
                    if (config_values.at("draw_func_id").is_number_integer()) {
                        config_values.at("draw_func_id").get_to(tmp_config.draw_func_id);
                    }
                    else {
                        check = false;
                    }
                    // hotkey
                    if (config_values.at("hotkey").is_array() && (config_values.at("hotkey").size() == 2)) {
                        if (config_values.at("hotkey")[0].is_number_integer() && config_values.at("hotkey")[1].is_number_integer()) {
                            int key = config_values.at("hotkey")[0].get<int>();
                            int mods = config_values.at("hotkey")[1].get<int>();
                            tmp_config.hotkey = core::view::KeyCode((core::view::Key)key, (core::view::Modifiers)mods);
                        }
                        else {
                            check = false;
                        }
                    }
                    else {
                        check = false;
                    }
                    // profile_position
                    if (config_values.at("profile_position").is_array() && (config_values.at("profile_position").size() == 2)) {
                        if (config_values.at("profile_position")[0].is_number_float()) {
                            config_values.at("profile_position")[0].get_to(tmp_config.profile_position.x);
                        }
                        else {
                            check = false;
                        }
                        if (config_values.at("profile_position")[1].is_number_float()) {
                            config_values.at("profile_position")[1].get_to(tmp_config.profile_position.y);
                        }
                        else {
                            check = false;
                        }
                    }
                    else {
                        check = false;
                    }
                    // profile_size
                    if (config_values.at("profile_size").is_array() && (config_values.at("profile_size").size() == 2)) {
                        if (config_values.at("profile_size")[0].is_number_float()) {
                            config_values.at("profile_size")[0].get_to(tmp_config.profile_size.x);
                        }
                        else {
                            check = false;
                        }
                        if (config_values.at("profile_size")[1].is_number_float()) {
                            config_values.at("profile_size")[1].get_to(tmp_config.profile_size.y);
                        }
                        else {
                            check = false;
                        }
                    }
                    else {
                        check = false;
                    }
                    // soft_reset
                    if (config_values.at("soft_reset").is_boolean()) {
                        config_values.at("soft_reset").get_to(tmp_config.soft_reset);
                    }
                    else {
                        check = false;
                    }
                    // soft_reset_size
                    if (config_values.at("soft_reset_size").is_array() && (config_values.at("soft_reset_size").size() == 2)) {
                        if (config_values.at("soft_reset_size")[0].is_number_float()) {
                            config_values.at("soft_reset_size")[0].get_to(tmp_config.soft_reset_size.x);
                        }
                        else {
                            check = false;
                        }
                        if (config_values.at("soft_reset_size")[1].is_number_float()) {
                            config_values.at("soft_reset_size")[1].get_to(tmp_config.soft_reset_size.y);
                        }
                        else {
                            check = false;
                        }
                    }
                    else {
                        check = false;
                    }
                    // show_hotkeys
                    if (config_values.at("show_hotkeys").is_boolean()) {
                        config_values.at("show_hotkeys").get_to(tmp_config.show_hotkeys);
                    }
                    else {
                        check = false;
                    }
                    // param_modules
                    tmp_config.param_modules.clear();
                    if (config_values.at("param_modules").is_array()) {
                        size_t tmp_size = config_values.at("param_modules").size();
                        for (size_t i = 0; i < tmp_size; ++i) {
                            if (config_values.at("param_modules")[i].is_string()) {
                                tmp_config.param_modules.emplace_back(config_values.at("param_modules")[i].get<std::string>());
                            }
                            else {
                                check = false;
                            }
                        }
                    }
                    else {
                        check = false;
                    }
                }
                catch (...) {
                    vislib::sys::Log::DefaultLog.WriteError("[GUIWindowManager] Error reading profile '%s'", profile_name.c_str());
                    return false;
                }
                // profile_reset
                tmp_config.profile_reset = true;

                tmp_windows.emplace(window_name, tmp_config);
            }
            if (check) {
                this->windows.clear();
                this->windows = tmp_windows;
                vislib::sys::Log::DefaultLog.WriteInfo("[GUIWindowManager] Successfully loaded profile '%s'.", profile_name.c_str());
                return true;
            }
        }
    }

    vislib::sys::Log::DefaultLog.WriteError("[GUIWindowManager] Couldn't load profile '%s'.", profile_name.c_str());
    return false;
}

/**
 * GUIWindowManager::DeleteWindowConfigurationProfile
 */
bool GUIWindowManager::DeleteWindowConfigurationProfile(std::string profile_name) {

    if (this->profiles.erase(profile_name) > 0) {
        // Saving changes immediately to ini file.
        return this->saveWindowConfigurationFile(); 
    }

    return false;
}

/**
 * GUIWindowManager::SaveWindowConfigurationProfile
 */
bool GUIWindowManager::SaveWindowConfigurationProfile(std::string profile_name) {

    /// Existing profile with same name will be overwritten ...
    for (auto& w : this->windows) {
        std::string window_name = w.first;
        WindowConfiguration window_config = w.second;
        this->profiles[profile_name][window_name]["show"]             = window_config.show;
        this->profiles[profile_name][window_name]["flags"]            = (int)(window_config.flags);
        this->profiles[profile_name][window_name]["draw_func_id"]     = window_config.draw_func_id;
        this->profiles[profile_name][window_name]["hotkey"]           = { (int)(window_config.hotkey.Key()), window_config.hotkey.Modifiers().toInt() };
        ///this->profiles[profile_name][window_name]["profile_reset"]    = window_config.profile_reset; // Always true for a loaded profile
        this->profiles[profile_name][window_name]["profile_position"] = { window_config.profile_position.x, window_config.profile_position.y };
        this->profiles[profile_name][window_name]["profile_size"]     = { window_config.profile_size.x, window_config.profile_size.y };
        this->profiles[profile_name][window_name]["soft_reset"]       = window_config.soft_reset;
        this->profiles[profile_name][window_name]["soft_reset_size"]  = { window_config.soft_reset_size.x, window_config.soft_reset_size.y };
        this->profiles[profile_name][window_name]["show_hotkeys"]     = window_config.show_hotkeys;
        this->profiles[profile_name][window_name]["param_modules"]    = window_config.param_modules;
    }

    // Saving changes immediately to ini file.
    return this->saveWindowConfigurationFile(); 
}


/**
 * GUIWindowManager::GetWindowConfigurationProfileList
 */
std::list<std::string> GUIWindowManager::GetWindowConfigurationProfileList(void) {

    std::list<std::string> out_list;

    for (auto &p : this->profiles.items()) {
        out_list.emplace_back(p.key());
    }

    return out_list;
}


/**
 * GUIWindowManager::saveWindowConfigurationFile
 */
bool GUIWindowManager::saveWindowConfigurationFile(void) {

    std::ofstream inifile;
    inifile.open(this->filename);

    /// Existing file with same name will be overwritten ...
    if (inifile.is_open() && inifile.good()) {
        inifile << this->profiles.dump(4);
        inifile.close();
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "[GUIWindowManager] Couldn't write to ini file: '%s'", this->filename.c_str());
    return false;
}


/**
 * GUIWindowManager::loadWindowConfigurationFile
 */
bool GUIWindowManager::loadWindowConfigurationFile(void) {

    std::ifstream inifile;
    inifile.open(this->filename);

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

        /// No check if loaded JSON contains only valid profiles ...
        this->profiles = parsed_json;
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "[GUIWindowManager] Couldn't read ini file: '%s'", this->filename.c_str());
    return false;
}

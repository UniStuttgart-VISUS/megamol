/*
 * WindowManager.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "WindowManager.h"


using namespace megamol::gui;


/**
 * WindowManager::Ctor
 */
WindowManager::WindowManager(std::string filename) : callbacks(), windows(), filename(filename) {

    // nothing to do here ...
}


/**
 * WindowManager::Dtor
 */
WindowManager::~WindowManager(void) { this->windows.clear(); }


/**
 * WindowManager::SoftResetWindowSizePos
 */
void WindowManager::SoftResetWindowSizePos(const std::string& window_name, WindowConfiguration& window_config) {

    assert(ImGui::GetCurrentContext() != nullptr);

    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    float width = window_config.win_reset_size.x;
    float height = window_config.win_reset_size.y;

    auto win_pos = ImGui::GetWindowPos();
    if (win_pos.x < 0) {
        win_pos.x = style.DisplayWindowPadding.x;
    }
    if (win_pos.y < 0) {
        win_pos.y = style.DisplayWindowPadding.y;
    }

    ImVec2 win_size;
    if (window_config.win_flags | ImGuiWindowFlags_AlwaysAutoResize) {
        win_size = ImGui::GetWindowSize();
    } else {
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
 * WindowManager::ResetWindowOnProfileLoad
 */
void WindowManager::ResetWindowOnProfileLoad(const std::string& window_name, WindowConfiguration& window_config) {

    assert(ImGui::GetCurrentContext() != nullptr);

    ImVec2 pos = window_config.win_position;
    ImVec2 size = window_config.win_size;

    ImGui::SetWindowSize(window_name.c_str(), size, ImGuiCond_Always);
    ImGui::SetWindowPos(window_name.c_str(), pos, ImGuiCond_Always);
}


/**
 * WindowManager::AddWindowConfiguration
 */
bool WindowManager::AddWindowConfiguration(const std::string& window_name, WindowConfiguration& window_config) {

    if (window_name.empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[WindowManager] No valid window name given.");
        return false;
    }
    if (this->windowConfigurationExists(window_name)) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[WindowManager] Found already existing window with name '%s'. Window names must be unique.",
            window_name.c_str());
        return false;
    }
    this->windows.emplace(window_name, window_config);
    return true;
}


/**
 * WindowManager::DeleteWindowConfiguration
 */
bool WindowManager::DeleteWindowConfiguration(const std::string& window_name) {

    if (!this->windowConfigurationExists(window_name)) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[WindowManager] Found no existing window with name '%s'.", window_name.c_str());
        return false;
    }
    this->windows.erase(window_name);
    return true;
}


/**
 * WindowManager::LoadWindowConfigurationProfile
 */
bool WindowManager::LoadWindowConfigurationProfile(const std::string& profile_name) {

    if (profile_name.empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[WindowManager] No valid profile name given.");
        return false;
    }

    bool check = true;
    std::map<std::string, WindowConfiguration> tmp_windows;

    nlohmann::json current_profiles;
    this->loadWindowConfigurationFile(current_profiles);

    for (auto& p : current_profiles.items()) {
        // Search for profile
        if (p.key() == profile_name) {
            // Loop over all windows
            for (auto& w : p.value().items()) {
                std::string window_name = w.key();
                WindowConfiguration tmp_config;
                // Getting all configuration values for current window.
                try {
                    auto config_values = w.value();

                    // WindowConfiguration ------------------------------------
                    // show
                    if (config_values.at("win_show").is_boolean()) {
                        config_values.at("win_show").get_to(tmp_config.win_show);
                    } else {
                        check = false;
                    }
                    // flags
                    if (config_values.at("win_flags").is_number_integer()) {
                        tmp_config.win_flags = (ImGuiWindowFlags)config_values.at("win_flags").get<int>();
                    } else {
                        check = false;
                    }
                    // callback
                    if (config_values.at("win_callback").is_number_integer()) {
                        tmp_config.win_callback = (WindowDrawCallback)config_values.at("win_callback").get<int>();
                    } else {
                        check = false;
                    }
                    // hotkey
                    if (config_values.at("win_hotkey").is_array() && (config_values.at("win_hotkey").size() == 2)) {
                        if (config_values.at("win_hotkey")[0].is_number_integer() &&
                            config_values.at("win_hotkey")[1].is_number_integer()) {
                            int key = config_values.at("win_hotkey")[0].get<int>();
                            int mods = config_values.at("win_hotkey")[1].get<int>();
                            tmp_config.win_hotkey =
                                core::view::KeyCode((core::view::Key)key, (core::view::Modifiers)mods);
                        } else {
                            check = false;
                        }
                    } else {
                        check = false;
                    }
                    // position
                    if (config_values.at("win_position").is_array() && (config_values.at("win_position").size() == 2)) {
                        if (config_values.at("win_position")[0].is_number_float()) {
                            config_values.at("win_position")[0].get_to(tmp_config.win_position.x);
                        } else {
                            check = false;
                        }
                        if (config_values.at("win_position")[1].is_number_float()) {
                            config_values.at("win_position")[1].get_to(tmp_config.win_position.y);
                        } else {
                            check = false;
                        }
                    } else {
                        check = false;
                    }
                    // size
                    if (config_values.at("win_size").is_array() && (config_values.at("win_size").size() == 2)) {
                        if (config_values.at("win_size")[0].is_number_float()) {
                            config_values.at("win_size")[0].get_to(tmp_config.win_size.x);
                        } else {
                            check = false;
                        }
                        if (config_values.at("win_size")[1].is_number_float()) {
                            config_values.at("win_size")[1].get_to(tmp_config.win_size.y);
                        } else {
                            check = false;
                        }
                    } else {
                        check = false;
                    }
                    // soft_reset
                    if (config_values.at("win_soft_reset").is_boolean()) {
                        config_values.at("win_soft_reset").get_to(tmp_config.win_soft_reset);
                    } else {
                        check = false;
                    }
                    // reset_size
                    if (config_values.at("win_reset_size").is_array() &&
                        (config_values.at("win_reset_size").size() == 2)) {
                        if (config_values.at("win_reset_size")[0].is_number_float()) {
                            config_values.at("win_reset_size")[0].get_to(tmp_config.win_reset_size.x);
                        } else {
                            check = false;
                        }
                        if (config_values.at("win_reset_size")[1].is_number_float()) {
                            config_values.at("win_reset_size")[1].get_to(tmp_config.win_reset_size.y);
                        } else {
                            check = false;
                        }
                    } else {
                        check = false;
                    }
                    // ParamConfig --------------------------------------------
                    // show_hotkeys
                    if (config_values.at("param_show_hotkeys").is_boolean()) {
                        config_values.at("param_show_hotkeys").get_to(tmp_config.param_show_hotkeys);
                    } else {
                        check = false;
                    }
                    // modules_list
                    tmp_config.param_modules_list.clear();
                    if (config_values.at("param_modules_list").is_array()) {
                        size_t tmp_size = config_values.at("param_modules_list").size();
                        for (size_t i = 0; i < tmp_size; ++i) {
                            if (config_values.at("param_modules_list")[i].is_string()) {
                                tmp_config.param_modules_list.emplace_back(
                                    config_values.at("param_modules_list")[i].get<std::string>());
                            } else {
                                check = false;
                            }
                        }
                    } else {
                        check = false;
                    }
                    // module_filter
                    if (config_values.at("param_module_filter").is_number_integer()) {
                        tmp_config.param_module_filter = (FilterMode)config_values.at("param_module_filter").get<int>();
                    } else {
                        check = false;
                    }
                    // FpsMsConfig --------------------------------------------
                    // show_options
                    if (config_values.at("fpsms_show_options").is_boolean()) {
                        config_values.at("fpsms_show_options").get_to(tmp_config.fpsms_show_options);
                    } else {
                        check = false;
                    }
                    // max_value_count
                    if (config_values.at("fpsms_max_value_count").is_number_integer()) {
                        config_values.at("fpsms_max_value_count").get_to(tmp_config.fpsms_max_value_count);
                    } else {
                        check = false;
                    }
                    // max_delay
                    if (config_values.at("fpsms_max_delay").is_number_float()) {
                        config_values.at("fpsms_max_delay").get_to(tmp_config.fpsms_max_delay);
                    } else {
                        check = false;
                    }
                    // mode
                    if (config_values.at("fpsms_mode").is_number_integer()) {
                        tmp_config.fpsms_mode = (FpsMsMode)config_values.at("fpsms_mode").get<int>();
                    } else {
                        check = false;
                    }
                    // FontConfig ---------------------------------------------
                    // font_name
                    if (config_values.at("font_name").is_string()) {
                        config_values.at("font_name").get_to(tmp_config.font_name);
                    } else {
                        check = false;
                    }
                } catch (...) {
                    vislib::sys::Log::DefaultLog.WriteError(
                        "[WindowManager] Unable to reading JSON of profile '%s'", profile_name.c_str());
                    return false;
                }
                // profile reset flags
                tmp_config.win_reset = true;
                tmp_config.font_reset = false;
                if (!tmp_config.font_name.empty()) {
                    tmp_config.font_reset = true;
                }

                tmp_windows.emplace(window_name, tmp_config);
            }
            if (check) {
                this->windows.clear();
                this->windows = tmp_windows;
                vislib::sys::Log::DefaultLog.WriteInfo(
                    "[WindowManager] Successfully loaded profile '%s'.", profile_name.c_str());
                return true;
            }
        }
    }

    vislib::sys::Log::DefaultLog.WriteWarn("[WindowManager] Couldn't load profile '%s'.", profile_name.c_str());
    return false;
}

/**
 * WindowManager::DeleteWindowConfigurationProfile
 */
bool WindowManager::DeleteWindowConfigurationProfile(const std::string& profile_name) {

    if (profile_name.empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[WindowManager] No valid profile name given.");
        return false;
    }

    nlohmann::json current_profiles;
    this->loadWindowConfigurationFile(current_profiles);
    if (current_profiles.erase(profile_name) != 0) {
        return this->saveWindowConfigurationFile(current_profiles);
        ;
    }

    return false;
}

/**
 * WindowManager::SaveWindowConfigurationProfile
 */
bool WindowManager::SaveWindowConfigurationProfile(const std::string& profile_name) {

    if (profile_name.empty()) {
        vislib::sys::Log::DefaultLog.WriteWarn("[WindowManager] No valid profile name given.");
        return false;
    }

    nlohmann::json current_profiles;
    this->loadWindowConfigurationFile(current_profiles);

    /// Existing profile with same name will be overwritten/merged with existing ...
    for (auto& w : this->windows) {
        std::string window_name = w.first;
        WindowConfiguration window_config = w.second;
        current_profiles[profile_name][window_name]["win_show"] = window_config.win_show;
        current_profiles[profile_name][window_name]["win_flags"] = (int)(window_config.win_flags);
        current_profiles[profile_name][window_name]["win_callback"] = window_config.win_callback;
        current_profiles[profile_name][window_name]["win_hotkey"] = {
            (int)(window_config.win_hotkey.GetKey()), window_config.win_hotkey.GetModifiers().toInt()};
        current_profiles[profile_name][window_name]["win_position"] = {
            window_config.win_position.x, window_config.win_position.y};
        current_profiles[profile_name][window_name]["win_size"] = {window_config.win_size.x, window_config.win_size.y};
        current_profiles[profile_name][window_name]["win_soft_reset"] = window_config.win_soft_reset;
        current_profiles[profile_name][window_name]["win_reset_size"] = {
            window_config.win_reset_size.x, window_config.win_reset_size.y};

        current_profiles[profile_name][window_name]["param_show_hotkeys"] = window_config.param_show_hotkeys;
        current_profiles[profile_name][window_name]["param_modules_list"] = window_config.param_modules_list;
        current_profiles[profile_name][window_name]["param_module_filter"] = window_config.param_module_filter;

        current_profiles[profile_name][window_name]["fpsms_show_options"] = window_config.fpsms_show_options;
        current_profiles[profile_name][window_name]["fpsms_max_value_count"] = window_config.fpsms_max_value_count;
        current_profiles[profile_name][window_name]["fpsms_max_delay"] = window_config.fpsms_max_delay;
        current_profiles[profile_name][window_name]["fpsms_mode"] = (int)window_config.fpsms_mode;

        current_profiles[profile_name][window_name]["font_name"] = window_config.font_name;
    }

    return this->saveWindowConfigurationFile(current_profiles);
}


/**
 * WindowManager::GetWindowConfigurationProfileList
 */
std::list<std::string> WindowManager::GetWindowConfigurationProfileList(void) {

    std::list<std::string> out_list;
    nlohmann::json current_profiles;
    if (this->loadWindowConfigurationFile(current_profiles)) {
        for (auto& p : current_profiles.items()) {
            out_list.emplace_back(p.key());
        }
    }

    return out_list;
}


/**
 * WindowManager::saveWindowConfigurationFile
 */
bool WindowManager::saveWindowConfigurationFile(nlohmann::json& in_profiles) {

    std::ofstream profilefile;
    profilefile.open(this->filename);

    /// Existing file with same name will be overwritten ...
    if (profilefile.is_open() && profilefile.good()) {
        profilefile << in_profiles.dump(4);
        profilefile.close();

        return true;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "[WindowManager] Couldn't write profile to file: '%s'", this->filename.c_str());
    return false;
}


/**
 * WindowManager::loadWindowConfigurationFile
 */
bool WindowManager::loadWindowConfigurationFile(nlohmann::json& out_profiles) {

    std::ifstream profilefile;
    profilefile.open(this->filename);

    std::string line;
    std::stringstream stream;

    if (profilefile.is_open() && profilefile.good()) {
        while (std::getline(profilefile, line)) {
            stream << line << std::endl;
        }
        profilefile.close();

        nlohmann::json parsed_json = nlohmann::json::parse(stream.str());

        // Check for valid JSON object
        if (!parsed_json.is_object()) {
            vislib::sys::Log::DefaultLog.WriteError("[WindowManager] Profile file content is no valid JSON object.");
            return false;
        }

        /// No check if loaded JSON contains only valid profiles ...
        out_profiles = parsed_json;
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "[WindowManager] Couldn't read profile from file: '%s'", this->filename.c_str());
    return false;
}

/*
 * WindowManager.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "WindowManager.h"

#include <sstream>


using namespace megamol::gui;


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


void WindowManager::ResetWindowOnStateLoad(const std::string& window_name, WindowConfiguration& window_config) {
    assert(ImGui::GetCurrentContext() != nullptr);

    ImVec2 pos = window_config.win_position;
    ImVec2 size = window_config.win_size;

    ImGui::SetWindowSize(window_name.c_str(), size, ImGuiCond_Always);
    ImGui::SetWindowPos(window_name.c_str(), pos, ImGuiCond_Always);
}


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


bool WindowManager::DeleteWindowConfiguration(const std::string& window_name) {
    if (!this->windowConfigurationExists(window_name)) {
        vislib::sys::Log::DefaultLog.WriteWarn(
            "[WindowManager] Found no existing window with name '%s'.", window_name.c_str());
        return false;
    }
    this->windows.erase(window_name);
    return true;
}


bool WindowManager::StateFromJSON(const std::string& json_string) {

    nlohmann::json json;
    try {
        json = nlohmann::json::parse(json_string);
    } catch (...) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[WindowManager] Unable to parse JSON string (there should be no escaped quotes, e.g.).");
        return false;
    }

    if (!json.is_object()) {
        vislib::sys::Log::DefaultLog.WriteError("[WindowManager] State has to be a valid JSON object.");
        return false;
    }

    bool valid = true;
    std::map<std::string, WindowConfiguration> tmp_windows;
    for (auto& w : json.items()) {
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
                valid = false;
            }
            // flags
            if (config_values.at("win_flags").is_number_integer()) {
                tmp_config.win_flags = (ImGuiWindowFlags)config_values.at("win_flags").get<int>();
            } else {
                valid = false;
            }
            // callback
            if (config_values.at("win_callback").is_number_integer()) {
                tmp_config.win_callback = (WindowDrawCallback)config_values.at("win_callback").get<int>();
            } else {
                valid = false;
            }
            // hotkey
            if (config_values.at("win_hotkey").is_array() && (config_values.at("win_hotkey").size() == 2)) {
                if (config_values.at("win_hotkey")[0].is_number_integer() &&
                    config_values.at("win_hotkey")[1].is_number_integer()) {
                    int key = config_values.at("win_hotkey")[0].get<int>();
                    int mods = config_values.at("win_hotkey")[1].get<int>();
                    tmp_config.win_hotkey = core::view::KeyCode((core::view::Key)key, (core::view::Modifiers)mods);
                } else {
                    valid = false;
                }
            } else {
                valid = false;
            }
            // position
            if (config_values.at("win_position").is_array() && (config_values.at("win_position").size() == 2)) {
                if (config_values.at("win_position")[0].is_number_float()) {
                    config_values.at("win_position")[0].get_to(tmp_config.win_position.x);
                } else {
                    valid = false;
                }
                if (config_values.at("win_position")[1].is_number_float()) {
                    config_values.at("win_position")[1].get_to(tmp_config.win_position.y);
                } else {
                    valid = false;
                }
            } else {
                valid = false;
            }
            // size
            if (config_values.at("win_size").is_array() && (config_values.at("win_size").size() == 2)) {
                if (config_values.at("win_size")[0].is_number_float()) {
                    config_values.at("win_size")[0].get_to(tmp_config.win_size.x);
                } else {
                    valid = false;
                }
                if (config_values.at("win_size")[1].is_number_float()) {
                    config_values.at("win_size")[1].get_to(tmp_config.win_size.y);
                } else {
                    valid = false;
                }
            } else {
                valid = false;
            }
            // soft_reset
            if (config_values.at("win_soft_reset").is_boolean()) {
                config_values.at("win_soft_reset").get_to(tmp_config.win_soft_reset);
            } else {
                valid = false;
            }
            // reset_size
            if (config_values.at("win_reset_size").is_array() && (config_values.at("win_reset_size").size() == 2)) {
                if (config_values.at("win_reset_size")[0].is_number_float()) {
                    config_values.at("win_reset_size")[0].get_to(tmp_config.win_reset_size.x);
                } else {
                    valid = false;
                }
                if (config_values.at("win_reset_size")[1].is_number_float()) {
                    config_values.at("win_reset_size")[1].get_to(tmp_config.win_reset_size.y);
                } else {
                    valid = false;
                }
            } else {
                valid = false;
            }
            // ParamConfig --------------------------------------------
            // show_hotkeys
            if (config_values.at("param_show_hotkeys").is_boolean()) {
                config_values.at("param_show_hotkeys").get_to(tmp_config.param_show_hotkeys);
            } else {
                valid = false;
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
                        valid = false;
                    }
                }
            } else {
                valid = false;
            }
            // module_filter
            if (config_values.at("param_module_filter").is_number_integer()) {
                tmp_config.param_module_filter = (FilterMode)config_values.at("param_module_filter").get<int>();
            } else {
                valid = false;
            }
            // FpsMsConfig --------------------------------------------
            // show_options
            if (config_values.at("fpsms_show_options").is_boolean()) {
                config_values.at("fpsms_show_options").get_to(tmp_config.fpsms_show_options);
            } else {
                valid = false;
            }
            // max_value_count
            if (config_values.at("fpsms_max_value_count").is_number_integer()) {
                config_values.at("fpsms_max_value_count").get_to(tmp_config.fpsms_max_value_count);
            } else {
                valid = false;
            }
            // max_delay
            if (config_values.at("fpsms_max_delay").is_number_float()) {
                config_values.at("fpsms_max_delay").get_to(tmp_config.fpsms_max_delay);
            } else {
                valid = false;
            }
            // mode
            if (config_values.at("fpsms_mode").is_number_integer()) {
                tmp_config.fpsms_mode = (TimingMode)config_values.at("fpsms_mode").get<int>();
            } else {
                valid = false;
            }
            // FontConfig ---------------------------------------------
            // font_name
            if (config_values.at("font_name").is_string()) {
                config_values.at("font_name").get_to(tmp_config.font_name);
            } else {
                valid = false;
            }
        } catch (...) {
            vislib::sys::Log::DefaultLog.WriteError("[WindowManager] Unable to reading JSON of state");
            return false;
        }
        // state reset flags
        tmp_config.win_reset = true;
        tmp_config.font_reset = false;
        if (!tmp_config.font_name.empty()) {
            tmp_config.font_reset = true;
        }

        tmp_windows.emplace(window_name, tmp_config);
    }

    if (!valid) {
        vislib::sys::Log::DefaultLog.WriteWarn("[WindowManager] Could not load state.");
        return false;
    }

    this->windows.clear();
    this->windows = tmp_windows;

    return true;
}


bool WindowManager::StateToJSON(std::string& json_string) {
    json_string = "";

    nlohmann::json json;
    for (auto& w : this->windows) {
        std::string window_name = w.first;
        WindowConfiguration window_config = w.second;
        json[window_name]["win_show"] = window_config.win_show;
        json[window_name]["win_flags"] = (int)(window_config.win_flags);
        json[window_name]["win_callback"] = window_config.win_callback;
        json[window_name]["win_hotkey"] = {
            (int)(window_config.win_hotkey.GetKey()), window_config.win_hotkey.GetModifiers().toInt()};
        json[window_name]["win_position"] = {window_config.win_position.x, window_config.win_position.y};
        json[window_name]["win_size"] = {window_config.win_size.x, window_config.win_size.y};
        json[window_name]["win_soft_reset"] = window_config.win_soft_reset;
        json[window_name]["win_reset_size"] = {window_config.win_reset_size.x, window_config.win_reset_size.y};

        json[window_name]["param_show_hotkeys"] = window_config.param_show_hotkeys;
        json[window_name]["param_modules_list"] = window_config.param_modules_list;
        json[window_name]["param_module_filter"] = window_config.param_module_filter;

        json[window_name]["fpsms_show_options"] = window_config.fpsms_show_options;
        json[window_name]["fpsms_max_value_count"] = window_config.fpsms_max_value_count;
        json[window_name]["fpsms_max_delay"] = window_config.fpsms_max_delay;
        json[window_name]["fpsms_mode"] = (int)window_config.fpsms_mode;

        json[window_name]["font_name"] = window_config.font_name;
    }

    std::stringstream ss;
    ss << json.dump(2);
    json_string = ss.str();

    return true;
}

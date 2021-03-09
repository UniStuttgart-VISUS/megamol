/*
 * WindowCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "WindowCollection.h"


using namespace megamol;
using namespace megamol::gui;


void WindowCollection::SetWindowSizePosition(WindowConfiguration& window_config, bool consider_menu) {

    assert(ImGui::GetCurrentContext() != nullptr);

    ImGuiIO& io = ImGui::GetIO();

    // Main menu height
    float y_offset = ImGui::GetFrameHeight();

    ImVec2 win_pos = window_config.win_position;
    ImVec2 win_size = window_config.win_size;
    if (window_config.win_flags & ImGuiWindowFlags_AlwaysAutoResize) {
        win_size = ImGui::GetWindowSize();
    }

    // Fit max window size to viewport
    if (win_size.x > io.DisplaySize.x) {
        win_size.x = io.DisplaySize.x;
    }
    if (win_size.y > (io.DisplaySize.y - y_offset)) {
        win_size.y = (io.DisplaySize.y - y_offset);
    }

    // Snap to viewport
    /// ImGui automatically moves windows lying outside viewport
    // float win_width = io.DisplaySize.x - (win_pos.x);
    // if (win_width < win_size.x) {
    //    win_pos.x = io.DisplaySize.x - (win_size.x);
    //}
    // float win_height = io.DisplaySize.y - (win_pos.y);
    // if (win_height < win_size.y) {
    //    win_pos.y = io.DisplaySize.y - (win_size.y);
    //}
    // if (win_pos.x < 0) {
    //    win_pos.x = 0.0f;
    //}

    // Snap window below menu bar
    if (consider_menu && (win_pos.y < y_offset)) {
        win_pos.y = y_offset;
    }

    window_config.win_position = win_pos;
    // window_config.win_reset_position = win_pos;
    ImGui::SetWindowPos(win_pos, ImGuiCond_Always);

    window_config.win_size = win_size;
    // window_config.win_reset_size = win_size;
    ImGui::SetWindowSize(win_size, ImGuiCond_Always);
}


bool WindowCollection::AddWindowConfiguration(WindowConfiguration& window_config) {
    if (window_config.win_name.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] No valid window name given. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->windowConfigurationExists(window_config.win_name)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Found already existing window with name '%s'. Window names must be unique. [%s, %s, line %d]\n",
            window_config.win_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->windows.emplace_back(window_config);
    return true;
}


bool WindowCollection::DeleteWindowConfiguration(const std::string& window_name) {
    if (!this->windowConfigurationExists(window_name)) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Could not find window with name '%s'. [%s, %s, line %d]\n", window_name.c_str(), __FILE__,
            __FUNCTION__, __LINE__);
        return false;
    }
    for (auto iter = this->windows.begin(); iter != this->windows.end(); iter++) {
        if (iter->win_name == window_name) {
            this->windows.erase(iter);
            return true;
        }
    }
    return false;
}


bool WindowCollection::StateFromJSON(const nlohmann::json& in_json) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        bool found = false;
        bool valid = true;
        std::vector<WindowConfiguration> tmp_windows;
        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
                found = true;
                for (auto& config_item : header_item.value().items()) {
                    WindowConfiguration tmp_config;
                    tmp_config.win_name = config_item.key();
                    tmp_config.win_set_pos_size = true;
                    tmp_config.buf_tfe_reset = true;
                    auto config_values = config_item.value();

                    // Window Configuration -----------------------------------
                    megamol::core::utility::get_json_value<bool>(config_values, {"win_show"}, &tmp_config.win_show);

                    int win_flags = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"win_flags"}, &win_flags);
                    tmp_config.win_flags = static_cast<ImGuiWindowFlags>(win_flags);

                    int win_callback = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"win_callback"}, &win_callback);
                    tmp_config.win_callback = static_cast<DrawCallbacks>(win_callback);

                    std::array<int, 2> hotkey = {0, 0};
                    megamol::core::utility::get_json_value<int>(
                        config_values, {"win_hotkey"}, hotkey.data(), hotkey.size());
                    tmp_config.win_hotkey = core::view::KeyCode(
                        static_cast<core::view::Key>(hotkey[0]), static_cast<core::view::Modifiers>(hotkey[1]));

                    std::array<float, 2> position;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_position"}, position.data(), position.size());
                    tmp_config.win_position = ImVec2(position[0], position[1]);

                    std::array<float, 2> size;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_size"}, size.data(), size.size());
                    tmp_config.win_size = ImVec2(size[0], size[1]);

                    std::array<float, 2> reset_size;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_reset_size"}, reset_size.data(), reset_size.size());
                    tmp_config.win_reset_size = ImVec2(reset_size[0], reset_size[1]);

                    std::array<float, 2> reset_position;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_reset_position"}, reset_position.data(), reset_position.size());
                    tmp_config.win_reset_position = ImVec2(reset_position[0], reset_position[1]);

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"win_collapsed"}, &tmp_config.win_collapsed);

                    // Param Config --------------------------------------------
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"param_show_hotkeys"}, &tmp_config.param_show_hotkeys);

                    tmp_config.param_modules_list.clear();
                    if (config_values.at("param_modules_list").is_array()) {
                        size_t buf_size = config_values.at("param_modules_list").size();
                        for (size_t i = 0; i < buf_size; ++i) {
                            std::string value;
                            megamol::core::utility::get_json_value<std::string>(
                                config_values.at("param_modules_list")[i], {}, &value);
                            tmp_config.param_modules_list.emplace_back(value);
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'param_modules_list' as array. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    int module_filter = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"param_module_filter"}, &module_filter);
                    tmp_config.param_module_filter = static_cast<FilterModes>(module_filter);

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"param_extended_mode"}, &tmp_config.param_extended_mode);

                    // FpsMs Config --------------------------------------------
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"ms_show_options"}, &tmp_config.ms_show_options);

                    megamol::core::utility::get_json_value<int>(
                        config_values, {"ms_max_history_count"}, &tmp_config.ms_max_history_count);

                    megamol::core::utility::get_json_value<float>(
                        config_values, {"ms_refresh_rate"}, &tmp_config.ms_refresh_rate);

                    int mode = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"ms_mode"}, &mode);
                    tmp_config.ms_mode = static_cast<TimingModes>(mode);

                    // TFE Config ---------------------------------------------
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"tfe_view_minimized"}, &tmp_config.tfe_view_minimized);

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"tfe_view_vertical"}, &tmp_config.tfe_view_vertical);

                    megamol::core::utility::get_json_value<std::string>(
                        config_values, {"tfe_active_param"}, &tmp_config.tfe_active_param);

                    // Log Config ---------------------------------------------
                    megamol::core::utility::get_json_value<unsigned int>(
                        config_values, {"log_level"}, &tmp_config.log_level);

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"log_force_open"}, &tmp_config.log_force_open);

                    // --------------------------------------------------------
                    // add current window config to tmp window config list
                    tmp_windows.emplace_back(tmp_config);
                }
            }
        }
        if (found) {
            if (valid) {
#ifdef GUI_VERBOSE
                megamol::core::utility::log::Log::DefaultLog.WriteInfo(
                    "[GUI] Read window configurations from JSON string.");
#endif // GUI_VERBOSE
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[GUI] Error while loading window configuration state from JSON. [%s, %s, line %d]\n", __FILE__,
                    __FUNCTION__, __LINE__);
                return false;
            }
        } else {
            return false;
        }

        /// Not omitting complete read windows configuration if there was an error while reading
        /// (preventing 'old' configurations going out of use)
        // Replace existing window configurations and add new windows.
        for (auto& new_win : tmp_windows) {
            bool found_existing = false;
            for (auto& win : this->windows) {
                // Check for same name
                if (win.win_name == new_win.win_name) {
                    win = new_win;
                    found_existing = true;
                }
            }
            if (!found_existing) {
                this->windows.emplace_back(new_win);
            }
        }
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to read state from JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool WindowCollection::StateToJSON(nlohmann::json& inout_json) {

    try {
        // Append to given json
        for (auto& window : this->windows) {
            if (window.win_store_config) {
                std::string window_name = window.win_name;
                WindowConfiguration window_config = window;

                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_show"] = window_config.win_show;
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_flags"] =
                    static_cast<int>(window_config.win_flags);
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_callback"] =
                    static_cast<int>(window_config.win_callback);
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_hotkey"] = {
                    static_cast<int>(window_config.win_hotkey.key), window_config.win_hotkey.mods.toInt()};
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_position"] = {
                    window_config.win_position.x, window_config.win_position.y};
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_size"] = {
                    window_config.win_size.x, window_config.win_size.y};
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_reset_size"] = {
                    window_config.win_reset_size.x, window_config.win_reset_size.y};
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_reset_position"] = {
                    window_config.win_reset_position.x, window_config.win_reset_position.y};
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_collapsed"] = window_config.win_collapsed;

                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["param_show_hotkeys"] =
                    window_config.param_show_hotkeys;

                for (auto& pm : window_config.param_modules_list) {
                    GUIUtils::Utf8Encode(pm);
                }
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["param_modules_list"] =
                    window_config.param_modules_list;
                for (auto& pm : window_config.param_modules_list) {
                    GUIUtils::Utf8Decode(pm);
                }

                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["param_module_filter"] =
                    static_cast<int>(window_config.param_module_filter);
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["param_extended_mode"] =
                    window_config.param_extended_mode;

                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["ms_show_options"] = window_config.ms_show_options;
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["ms_max_history_count"] =
                    window_config.ms_max_history_count;
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["ms_refresh_rate"] = window_config.ms_refresh_rate;
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["ms_mode"] =
                    static_cast<int>(window_config.ms_mode);

                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["tfe_view_minimized"] =
                    window_config.tfe_view_minimized;
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["tfe_view_vertical"] =
                    window_config.tfe_view_vertical;

                GUIUtils::Utf8Encode(window_config.tfe_active_param);
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["tfe_active_param"] =
                    window_config.tfe_active_param;
                GUIUtils::Utf8Decode(window_config.tfe_active_param);

                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["log_level"] = window_config.log_level;
                inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["log_force_open"] = window_config.log_force_open;
            }
        }
#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote window configurations to JSON.");
#endif // GUI_VERBOSE

    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON Error - Unable to write state to JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}

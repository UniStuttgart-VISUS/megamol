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


void WindowCollection::SoftResetWindowSizePosition(WindowConfiguration& window_config) {
    assert(ImGui::GetCurrentContext() != nullptr);

    ImGuiIO& io = ImGui::GetIO();

    float width = window_config.win_reset_size.x;
    float height = window_config.win_reset_size.y;

    if (width > io.DisplaySize.x) {
        width = io.DisplaySize.x;
    }
    if (height > io.DisplaySize.y) {
        height = io.DisplaySize.y;
    }

    auto win_pos = ImGui::GetWindowPos();
    if (win_pos.x < 0) {
        win_pos.x = 0.0f;
    }
    if (win_pos.y < 0) {
        win_pos.y = 0.0f;
    }

    ImVec2 win_size;
    if (window_config.win_flags & ImGuiWindowFlags_AlwaysAutoResize) {
        win_size = ImGui::GetWindowSize();
    } else {
        win_size = ImVec2(width, height);
    }

    float win_width = io.DisplaySize.x - (win_pos.x);
    if (win_width < win_size.x) {
        win_pos.x = io.DisplaySize.x - (win_size.x);
    }
    float win_height = io.DisplaySize.y - (win_pos.y);
    if (win_height < win_size.y) {
        win_pos.y = io.DisplaySize.y - (win_size.y);
    }

    ImGui::SetWindowSize(ImVec2(width, height), ImGuiCond_Always);
    ImGui::SetWindowPos(win_pos, ImGuiCond_Always);
}


void WindowCollection::ResetWindowSizePosition(WindowConfiguration& window_config) {
    assert(ImGui::GetCurrentContext() != nullptr);

    ImVec2 pos = window_config.win_position;
    ImVec2 size = window_config.win_size;

    ImGui::SetWindowSize(size, ImGuiCond_Always);
    ImGui::SetWindowPos(pos, ImGuiCond_Always);
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


bool WindowCollection::StateFromJsonString(const std::string& in_json_string) {

    try {
        if (in_json_string.empty()) {
            return false;
        }

        bool found = false;
        bool valid = true;
        std::vector<WindowConfiguration> tmp_windows;

        nlohmann::json json;
        json = nlohmann::json::parse(in_json_string);

        if (!json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] State is no valid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        for (auto& header_item : json.items()) {
            if (header_item.key() == (GUI_JSON_TAG_WINDOW_CONFIGURATIONS)) {
                found = true;
                for (auto& config_item : header_item.value().items()) {
                    WindowConfiguration tmp_config;
                    tmp_config.win_name = config_item.key();
                    tmp_config.win_reset = true;
                    tmp_config.buf_font_reset = false;
                    tmp_config.buf_tfe_reset = false;

                    // Getting all configuration values for current window.
                    auto config_values = config_item.value();

                    // WindowConfiguration ------------------------------------
                    // show
                    if (config_values.at("win_show").is_boolean()) {
                        config_values.at("win_show").get_to(tmp_config.win_show);
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_show' as boolean. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // flags
                    if (config_values.at("win_flags").is_number_integer()) {
                        tmp_config.win_flags = static_cast<ImGuiWindowFlags>(config_values.at("win_flags").get<int>());
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_flags' as integer. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // callback
                    if (config_values.at("win_callback").is_number_integer()) {
                        tmp_config.win_callback =
                            static_cast<DrawCallbacks>(config_values.at("win_callback").get<int>());
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_callback' as integer. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // hotkey
                    if (config_values.at("win_hotkey").is_array() && (config_values.at("win_hotkey").size() == 2)) {
                        if (config_values.at("win_hotkey")[0].is_number_integer() &&
                            config_values.at("win_hotkey")[1].is_number_integer()) {
                            int key = config_values.at("win_hotkey")[0].get<int>();
                            int mods = config_values.at("win_hotkey")[1].get<int>();
                            tmp_config.win_hotkey = core::view::KeyCode(
                                static_cast<core::view::Key>(key), static_cast<core::view::Modifiers>(mods));
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read 'win_hotkey' values as integers. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_hotkey' as array of size two. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // position
                    if (config_values.at("win_position").is_array() && (config_values.at("win_position").size() == 2)) {
                        if (config_values.at("win_position")[0].is_number_float()) {
                            config_values.at("win_position")[0].get_to(tmp_config.win_position.x);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read first value of "
                                "'win_position' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                        if (config_values.at("win_position")[1].is_number_float()) {
                            config_values.at("win_position")[1].get_to(tmp_config.win_position.y);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read second value of "
                                "'win_position' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_position' as array of size two. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // size
                    if (config_values.at("win_size").is_array() && (config_values.at("win_size").size() == 2)) {
                        if (config_values.at("win_size")[0].is_number_float()) {
                            config_values.at("win_size")[0].get_to(tmp_config.win_size.x);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read first value of 'win_size' as float. [%s, %s, line "
                                "%d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                        if (config_values.at("win_size")[1].is_number_float()) {
                            config_values.at("win_size")[1].get_to(tmp_config.win_size.y);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read second value of 'win_size' as float. [%s, %s, line "
                                "%d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_size' as array of size two. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // soft_reset
                    if (config_values.at("win_soft_reset").is_boolean()) {
                        config_values.at("win_soft_reset").get_to(tmp_config.win_soft_reset);
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_soft_reset' as boolean. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // reset_size
                    if (config_values.at("win_reset_size").is_array() &&
                        (config_values.at("win_reset_size").size() == 2)) {
                        if (config_values.at("win_reset_size")[0].is_number_float()) {
                            config_values.at("win_reset_size")[0].get_to(tmp_config.win_reset_size.x);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read first value of "
                                "'win_reset_size' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                        if (config_values.at("win_reset_size")[1].is_number_float()) {
                            config_values.at("win_reset_size")[1].get_to(tmp_config.win_reset_size.y);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read second value  of "
                                "'win_reset_size' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_reset_size' as array of size two. [%s, %s, line "
                            "%d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // reset_position
                    if (config_values.at("win_reset_position").is_array() &&
                        (config_values.at("win_reset_position").size() == 2)) {
                        if (config_values.at("win_reset_position")[0].is_number_float()) {
                            config_values.at("win_reset_position")[0].get_to(tmp_config.win_reset_position.x);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read first value of "
                                "'win_reset_position' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                        if (config_values.at("win_reset_position")[1].is_number_float()) {
                            config_values.at("win_reset_position")[1].get_to(tmp_config.win_reset_position.y);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] JSON state: Failed to read second value  of "
                                "'win_reset_position' as float. [%s, %s, line %d]\n",
                                __FILE__, __FUNCTION__, __LINE__);
                            valid = false;
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'win_reset_position' as array of size two. [%s, %s, line "
                            "%d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // ParamConfig --------------------------------------------
                    // show_hotkeys
                    if (config_values.at("param_show_hotkeys").is_boolean()) {
                        config_values.at("param_show_hotkeys").get_to(tmp_config.param_show_hotkeys);
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'param_show_hotkeys' as boolean. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // modules_list (no UTF-8 support needed)
                    tmp_config.param_modules_list.clear();
                    if (config_values.at("param_modules_list").is_array()) {
                        size_t buf_size = config_values.at("param_modules_list").size();
                        for (size_t i = 0; i < buf_size; ++i) {
                            if (config_values.at("param_modules_list")[i].is_string()) {
                                tmp_config.param_modules_list.emplace_back(
                                    config_values.at("param_modules_list")[i].get<std::string>());
                            } else {
                                megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[GUI] JSON state: Failed to read element of 'param_modules_list' as string. [%s, "
                                    "%s, "
                                    "line %d]\n",
                                    __FILE__, __FUNCTION__, __LINE__);
                                valid = false;
                            }
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'param_modules_list' as array. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // module_filter
                    if (config_values.at("param_module_filter").is_number_integer()) {
                        tmp_config.param_module_filter =
                            static_cast<FilterModes>(config_values.at("param_module_filter").get<int>());
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'param_module_filter' as integer. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // extended_mode
                    if (config_values.at("param_extended_mode").is_boolean()) {
                        config_values.at("param_extended_mode").get_to(tmp_config.param_extended_mode);
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'param_extended_mode' as boolean. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    // FpsMsConfig --------------------------------------------
                    // show_options
                    if (config_values.at("ms_show_options").is_boolean()) {
                        config_values.at("ms_show_options").get_to(tmp_config.ms_show_options);
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'ms_show_options' as boolean. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // max_value_count
                    if (config_values.at("ms_max_history_count").is_number_integer()) {
                        config_values.at("ms_max_history_count").get_to(tmp_config.ms_max_history_count);
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'ms_max_history_count' as integer. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // max_delay
                    if (config_values.at("ms_refresh_rate").is_number_float()) {
                        config_values.at("ms_refresh_rate").get_to(tmp_config.ms_refresh_rate);
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'ms_refresh_rate' as float. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // mode
                    if (config_values.at("ms_mode").is_number_integer()) {
                        tmp_config.ms_mode = static_cast<TimingModes>(config_values.at("ms_mode").get<int>());
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'ms_mode' as integer. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // FontConfig ---------------------------------------------
                    // font_name (supports UTF-8)
                    if (config_values.at("font_name").is_string()) {
                        config_values.at("font_name").get_to(tmp_config.font_name);
                        GUIUtils::Utf8Decode(tmp_config.font_name);

                        if (!tmp_config.font_name.empty()) {
                            tmp_config.buf_font_reset = true;
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'font_name' as string. [%s, %s, line %d]\n", __FILE__,
                            __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // FTFEConfig ---------------------------------------------
                    // tfe_view_minimized
                    if (config_values.at("tfe_view_minimized").is_boolean()) {
                        config_values.at("tfe_view_minimized").get_to(tmp_config.tfe_view_minimized);

                        tmp_config.buf_tfe_reset = true;
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'tfe_view_minimized' as boolean. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // tfe_view_vertical
                    if (config_values.at("tfe_view_vertical").is_boolean()) {
                        config_values.at("tfe_view_vertical").get_to(tmp_config.tfe_view_vertical);

                        tmp_config.buf_tfe_reset = true;
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'tfe_view_vertical' as boolean. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }
                    // tfe_active_param (supports UTF-8)
                    if (config_values.at("tfe_active_param").is_string()) {
                        config_values.at("tfe_active_param").get_to(tmp_config.tfe_active_param);
                        GUIUtils::Utf8Decode(tmp_config.tfe_active_param);

                        if (!tmp_config.font_name.empty()) {
                            tmp_config.buf_tfe_reset = true;
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'tfe_active_param' as string. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }

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
        } else { // !found
            /// megamol::core::utility::log::Log::DefaultLog.WriteWarn("[GUI] Could not find window configuration state
            /// in JSON. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
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

    } catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error - Unable to parse JSON string. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }

    return true;
}


bool WindowCollection::StateToJSON(nlohmann::json& out_json) {

    try {
        /// Append to given json
        // out_json.clear();

        for (auto& window : this->windows) {
            if (window.win_store_config) {
                std::string window_name = window.win_name;
                WindowConfiguration window_config = window;

                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_show"] = window_config.win_show;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_flags"] =
                    static_cast<int>(window_config.win_flags);
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_callback"] =
                    static_cast<int>(window_config.win_callback);
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_hotkey"] = {
                    static_cast<int>(window_config.win_hotkey.key), window_config.win_hotkey.mods.toInt()};
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_position"] = {
                    window_config.win_position.x, window_config.win_position.y};
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_size"] = {
                    window_config.win_size.x, window_config.win_size.y};
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_soft_reset"] =
                    window_config.win_soft_reset;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_reset_size"] = {
                    window_config.win_reset_size.x, window_config.win_reset_size.y};
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["win_reset_position"] = {
                    window_config.win_reset_position.x, window_config.win_reset_position.y};

                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["param_show_hotkeys"] =
                    window_config.param_show_hotkeys;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["param_modules_list"] =
                    window_config.param_modules_list;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["param_module_filter"] =
                    static_cast<int>(window_config.param_module_filter);
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["param_extended_mode"] =
                    window_config.param_extended_mode;

                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["ms_show_options"] =
                    window_config.ms_show_options;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["ms_max_history_count"] =
                    window_config.ms_max_history_count;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["ms_refresh_rate"] =
                    window_config.ms_refresh_rate;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["ms_mode"] =
                    static_cast<int>(window_config.ms_mode);

                GUIUtils::Utf8Encode(window_config.font_name);
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["font_name"] = window_config.font_name;

                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["tfe_view_minimized"] =
                    window_config.tfe_view_minimized;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["tfe_view_vertical"] =
                    window_config.tfe_view_vertical;
                out_json[GUI_JSON_TAG_WINDOW_CONFIGURATIONS][window_name]["tfe_active_param"] =
                    window_config.tfe_active_param;
            }
        }
#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Wrote window configurations to JSON.");
#endif // GUI_VERBOSE

    } catch (nlohmann::json::type_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::invalid_iterator& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::out_of_range& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (nlohmann::json::other_error& e) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] JSON ERROR - %s: %s (%s:%d)", __FUNCTION__, e.what(), __FILE__, __LINE__);
        return false;
    } catch (...) {
        megamol::core::utility::log::Log::DefaultLog.WriteError(
            "[GUI] Unknown Error - Unable to write JSON of state. [%s, %s, line %d]\n", __FILE__, __FUNCTION__,
            __LINE__);
        return false;
    }

    return true;
}

/*
 * WindowCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "WindowCollection.h"
#include "GUIUtils.h"
#include "imgui_internal.h"


using namespace megamol;
using namespace megamol::gui;


void WindowConfiguration::ApplyWindowSizePosition(bool consider_menu) {

    assert(ImGui::GetCurrentContext() != nullptr);

    ImGuiIO& io = ImGui::GetIO();

    // Main menu height
    float y_offset = ImGui::GetFrameHeight();

    ImVec2 win_pos = this->config.basic.position;
    ImVec2 win_size = this->config.basic.size;
    if (this->config.basic.flags & ImGuiWindowFlags_AlwaysAutoResize) {
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

    this->config.basic.position = win_pos;
    // wc.config.basic.reset_position = win_pos;
    ImGui::SetWindowPos(win_pos, ImGuiCond_Always);

    this->config.basic.size = win_size;
    // wc.config.basic.reset_size = win_size;
    ImGui::SetWindowSize(win_size, ImGuiCond_Always);
}


// --------------------------------------------------------------------


bool WindowCollection::AddWindowConfiguration(WindowConfiguration& wc) {
    if (wc.Name().empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Invalid window name. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    if (this->WindowConfigurationExists(wc.Hash())) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Found already existing window with name '%s'. Window names must be unique. [%s, %s, line %d]\n",
            wc.Name().c_str(), __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    this->windows.emplace_back(wc);
    return true;
}


bool WindowCollection::DeleteWindowConfiguration(size_t win_hash_id) {
    for (auto iter = this->windows.begin(); iter != this->windows.end(); iter++) {
        if (iter->Hash() == win_hash_id) {
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
                    auto config_values = config_item.value();

                    int win_callback_id = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"win_callback"}, &win_callback_id);

                    WindowConfiguration tmp_config(
                        config_item.key(), static_cast<WindowConfiguration::PredefinedCallbackID>(win_callback_id));

                    tmp_config.config.basic.reset_pos_size = true;
                    tmp_config.config.specific.tmp_tfe_reset = true;

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"win_show"}, &tmp_config.config.basic.show);

                    int win_flags = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"win_flags"}, &win_flags);
                    tmp_config.config.basic.flags = static_cast<ImGuiWindowFlags>(win_flags);

                    std::array<int, 2> hotkey = {0, 0};
                    megamol::core::utility::get_json_value<int>(
                        config_values, {"win_hotkey"}, hotkey.data(), hotkey.size());
                    tmp_config.config.basic.hotkey = core::view::KeyCode(
                        static_cast<core::view::Key>(hotkey[0]), static_cast<core::view::Modifiers>(hotkey[1]));

                    std::array<float, 2> position;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_position"}, position.data(), position.size());
                    tmp_config.config.basic.position = ImVec2(position[0], position[1]);

                    std::array<float, 2> size;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_size"}, size.data(), size.size());
                    tmp_config.config.basic.size = ImVec2(size[0], size[1]);

                    std::array<float, 2> reset_size;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_reset_size"}, reset_size.data(), reset_size.size());
                    tmp_config.config.basic.reset_size = ImVec2(reset_size[0], reset_size[1]);

                    std::array<float, 2> reset_position;
                    megamol::core::utility::get_json_value<float>(
                        config_values, {"win_reset_position"}, reset_position.data(), reset_position.size());
                    tmp_config.config.basic.reset_position = ImVec2(reset_position[0], reset_position[1]);

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"win_collapsed"}, &tmp_config.config.basic.collapsed);

                    // Param Config --------------------------------------------
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"param_show_hotkeys"}, &tmp_config.config.specific.param_show_hotkeys);

                    tmp_config.config.specific.param_modules_list.clear();
                    if (config_values.at("param_modules_list").is_array()) {
                        size_t tmp_size = config_values.at("param_modules_list").size();
                        for (size_t i = 0; i < tmp_size; ++i) {
                            std::string value;
                            megamol::core::utility::get_json_value<std::string>(
                                config_values.at("param_modules_list")[i], {}, &value);
                            tmp_config.config.specific.param_modules_list.emplace_back(value);
                        }
                    } else {
                        megamol::core::utility::log::Log::DefaultLog.WriteError(
                            "[GUI] JSON state: Failed to read 'param_modules_list' as array. [%s, %s, line %d]\n",
                            __FILE__, __FUNCTION__, __LINE__);
                        valid = false;
                    }

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"param_extended_mode"}, &tmp_config.config.specific.param_extended_mode);

                    // FpsMs Config --------------------------------------------
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"fpsms_show_options"}, &tmp_config.config.specific.fpsms_show_options);

                    megamol::core::utility::get_json_value<int>(
                        config_values, {"fpsms_max_value_count"}, &tmp_config.config.specific.fpsms_buffer_size);

                    megamol::core::utility::get_json_value<float>(
                        config_values, {"fpsms_refresh_rate"}, &tmp_config.config.specific.fpsms_refresh_rate);

                    int mode = 0;
                    megamol::core::utility::get_json_value<int>(config_values, {"fpsms_mode"}, &mode);
                    tmp_config.config.specific.fpsms_mode = static_cast<WindowConfiguration::TimingMode>(mode);

                    // TFE Config ---------------------------------------------
                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"tfe_view_minimized"}, &tmp_config.config.specific.tfe_view_minimized);

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"tfe_view_vertical"}, &tmp_config.config.specific.tfe_view_vertical);

                    megamol::core::utility::get_json_value<std::string>(
                        config_values, {"tfe_active_param"}, &tmp_config.config.specific.tfe_active_param);

                    // Log Config ---------------------------------------------
                    megamol::core::utility::get_json_value<unsigned int>(
                        config_values, {"log_level"}, &tmp_config.config.specific.log_level);

                    megamol::core::utility::get_json_value<bool>(
                        config_values, {"log_force_open"}, &tmp_config.config.specific.log_force_open);

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
                if (win.Hash() == new_win.Hash()) {
                    auto tmp_volatile_callback = win.VolatileCallback();
                    win = new_win;
                    if (tmp_volatile_callback) {
                        // Restore previously set volatile callback
                        win.SetVolatileCallback(tmp_volatile_callback);
                    }
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
            std::string window_name = window.Name();
            WindowConfiguration wc = window;

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_show"] = wc.config.basic.show;
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_flags"] = static_cast<int>(wc.config.basic.flags);
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_callback"] = static_cast<int>(wc.CallbackID());
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_hotkey"] = {
                static_cast<int>(wc.config.basic.hotkey.key), wc.config.basic.hotkey.mods.toInt()};
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_position"] = {
                wc.config.basic.position.x, wc.config.basic.position.y};

            auto rescale_win_size = wc.config.basic.size;
            rescale_win_size /= megamol::gui::gui_scaling.Get();
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_size"] = {rescale_win_size.x, rescale_win_size.y};

            auto rescale_win_reset_size = wc.config.basic.reset_size;
            rescale_win_reset_size /= megamol::gui::gui_scaling.Get();
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_reset_size"] = {
                rescale_win_reset_size.x, rescale_win_reset_size.y};

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_reset_position"] = {
                wc.config.basic.reset_position.x, wc.config.basic.reset_position.y};
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["win_collapsed"] = wc.config.basic.collapsed;

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["param_show_hotkeys"] =
                wc.config.specific.param_show_hotkeys;

            for (auto& pm : wc.config.specific.param_modules_list) {
                GUIUtils::Utf8Encode(pm);
            }
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["param_modules_list"] =
                wc.config.specific.param_modules_list;
            for (auto& pm : wc.config.specific.param_modules_list) {
                GUIUtils::Utf8Decode(pm);
            }

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["param_extended_mode"] =
                wc.config.specific.param_extended_mode;

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["fpsms_show_options"] =
                wc.config.specific.fpsms_show_options;
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["fpsms_max_value_count"] =
                wc.config.specific.fpsms_buffer_size;
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["fpsms_refresh_rate"] =
                wc.config.specific.fpsms_refresh_rate;
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["fpsms_mode"] =
                static_cast<int>(wc.config.specific.fpsms_mode);

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["tfe_view_minimized"] =
                wc.config.specific.tfe_view_minimized;
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["tfe_view_vertical"] =
                wc.config.specific.tfe_view_vertical;

            GUIUtils::Utf8Encode(wc.config.specific.tfe_active_param);
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["tfe_active_param"] =
                wc.config.specific.tfe_active_param;
            GUIUtils::Utf8Decode(wc.config.specific.tfe_active_param);

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["log_level"] = wc.config.specific.log_level;
            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window_name]["log_force_open"] = wc.config.specific.log_force_open;
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

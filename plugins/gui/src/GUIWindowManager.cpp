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
    callbacks()
    , windows()
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
 * GUIWindowManager::SoftResetWindowSizePos
 */
void GUIWindowManager::SoftResetWindowSizePos(const std::string& window_name, WindowConfiguration& window_config) {

    assert(ImGui::GetCurrentContext() != nullptr);
    ImGuiIO& io = ImGui::GetIO();
    ImGuiStyle& style = ImGui::GetStyle();

    float width = window_config.reset_size.x;
    float height = window_config.reset_size.y;

    auto win_pos = ImGui::GetWindowPos();
    if (win_pos.x < 0) {
        win_pos.x = style.DisplayWindowPadding.x;
    }
    if (win_pos.y < 0) {
        win_pos.y = style.DisplayWindowPadding.y;
    }

    ImVec2 win_size;
    if (window_config.flags | ImGuiWindowFlags_AlwaysAutoResize) {
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
 * GUIWindowManager::ResetWindowOnProfileLoad
 */
void GUIWindowManager::ResetWindowOnProfileLoad(const std::string& window_name, WindowConfiguration& window_config) {

    assert(ImGui::GetCurrentContext() != nullptr);

    ImVec2 pos = window_config.position;
    ImVec2 size = window_config.size;

    ImGui::SetWindowSize(window_name.c_str(), size, ImGuiCond_Always);
    ImGui::SetWindowPos(window_name.c_str(), pos, ImGuiCond_Always);
}


/**
 * GUIWindowManager::RegisterDrawWindowCallback
 */
bool GUIWindowManager::RegisterDrawWindowCallback(WindowDrawCallback cbid, GuiCallbackFunc cb) {

    this->callbacks[cbid] = cb;

    return true;
}


/**
 * GUIWindowManager::DrawWindowContent
 */
bool GUIWindowManager::DrawWindowContent(const std::string& window_name) {

    if (!this->windowConfigurationExists(window_name)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[GUIWindowManager] Found no existing window '%s'.", window_name.c_str());
        return false;
    }

    WindowDrawCallback cdid = this->windows[window_name].callback;
    if (this->callbacks.find(cdid) == this->callbacks.end()) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[GUIWindowManager] Found no registerd callback for WindowDrawCallback '%d'", (int)cdid);
        return false;
    }

    this->callbacks[cdid](window_name, this->windows[window_name]);

    return true;
}


/**
 * GUIWindowManager::AddWindowConfiguration
 */
bool GUIWindowManager::AddWindowConfiguration(const std::string& window_name, WindowConfiguration& window_config) {

    if (this->windowConfigurationExists(window_name)) {
        vislib::sys::Log::DefaultLog.WriteError(
            "[GUIWindowManager] Found already existing window '%s'. Window name must be unique.", window_name.c_str());
        return false;
    }
    this->windows.emplace(window_name, window_config);
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
 * GUIWindowManager::DeleteWindowConfiguration
 */
//bool GUIWindowManager::DeleteWindowConfiguration(const std::string& window_name) {
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
 * GUIWindowManager::LoadWindowConfigurationProfile
 */
bool GUIWindowManager::LoadWindowConfigurationProfile(const std::string& profile_name) {

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

                    // WindowConfiguration ------------------------------------
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
                    // callback
                    if (config_values.at("callback").is_number_integer()) {
                        tmp_config.callback = (WindowDrawCallback)config_values.at("callback").get<int>();
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
                    // position
                    if (config_values.at("position").is_array() && (config_values.at("position").size() == 2)) {
                        if (config_values.at("position")[0].is_number_float()) {
                            config_values.at("position")[0].get_to(tmp_config.position.x);
                        }
                        else {
                            check = false;
                        }
                        if (config_values.at("position")[1].is_number_float()) {
                            config_values.at("position")[1].get_to(tmp_config.position.y);
                        }
                        else {
                            check = false;
                        }
                    }
                    else {
                        check = false;
                    }
                    // size
                    if (config_values.at("size").is_array() && (config_values.at("size").size() == 2)) {
                        if (config_values.at("size")[0].is_number_float()) {
                            config_values.at("size")[0].get_to(tmp_config.size.x);
                        }
                        else {
                            check = false;
                        }
                        if (config_values.at("size")[1].is_number_float()) {
                            config_values.at("size")[1].get_to(tmp_config.size.y);
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
                    // reset_size
                    if (config_values.at("reset_size").is_array() && (config_values.at("reset_size").size() == 2)) {
                        if (config_values.at("reset_size")[0].is_number_float()) {
                            config_values.at("reset_size")[0].get_to(tmp_config.reset_size.x);
                        }
                        else {
                            check = false;
                        }
                        if (config_values.at("reset_size")[1].is_number_float()) {
                            config_values.at("reset_size")[1].get_to(tmp_config.reset_size.y);
                        }
                        else {
                            check = false;
                        }
                    }
                    else {
                        check = false;
                    }

                    // ParamConfig --------------------------------------------
                    auto param_config = config_values.at("param_config");
                    // show_hotkeys
                    if (param_config.at("show_hotkeys").is_boolean()) {
                        param_config.at("show_hotkeys").get_to(tmp_config.param_config.show_hotkeys);
                    }
                    else {
                        check = false;
                    }
                    // param_modules
                    tmp_config.param_config.modules_list.clear();
                    if (param_config.at("modules_list").is_array()) {
                        size_t tmp_size = param_config.at("modules_list").size();
                        for (size_t i = 0; i < tmp_size; ++i) {
                            if (param_config.at("modules_list")[i].is_string()) {
                                tmp_config.param_config.modules_list.emplace_back(param_config.at("modules_list")[i].get<std::string>());
                            }
                            else {
                                check = false;
                            }
                        }
                    }
                    else {
                        check = false;
                    }
                    // module_filter
                    if (param_config.at("module_filter").is_number_integer()) {
                        tmp_config.param_config.module_filter = (FilterMode)param_config.at("module_filter").get<int>();
                    }
                    else {
                        check = false;
                    }

                    // FpsMsConfig --------------------------------------------
                    auto fpsms_config = config_values.at("fpsms_config");
                    // show_hotkeys
                    if (fpsms_config.at("show_options").is_boolean()) {
                        fpsms_config.at("show_options").get_to(tmp_config.fpsms_config.show_options);
                    }
                    else {
                        check = false;
                    }
                    // max_value_count
                    if (fpsms_config.at("max_value_count").is_number_integer()) {
                        fpsms_config.at("max_value_count").get_to(tmp_config.fpsms_config.max_value_count);
                    }
                    else {
                        check = false;
                    }
                    // max_delay
                    if (fpsms_config.at("max_delay").is_number_float()) {
                        fpsms_config.at("max_delay").get_to(tmp_config.fpsms_config.max_delay);
                    }
                    else {
                        check = false;
                    }
                    // mode
                    if (fpsms_config.at("mode").is_number_integer()) {
                        tmp_config.fpsms_config.mode = (FpsMsMode)fpsms_config.at("mode").get<int>();
                    }
                    else {
                        check = false;
                    }

                    // FontConfig ---------------------------------------------
                    // font_name
                    auto font_config = config_values.at("font_config");
                    if (font_config.at("font_name").is_string()) {
                        font_config.at("font_name").get_to(tmp_config.font_config.font_name);
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
                tmp_config.reset = true;

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
bool GUIWindowManager::DeleteWindowConfigurationProfile(const std::string& profile_name) {

    if (this->profiles.erase(profile_name) > 0) {
        // Saving changes immediately to profile file.
        return this->saveWindowConfigurationFile(); 
    }

    return false;
}

/**
 * GUIWindowManager::SaveWindowConfigurationProfile
 */
bool GUIWindowManager::SaveWindowConfigurationProfile(const std::string& profile_name) {

    /// Existing profile with same name will be overwritten ...
    for (auto& w : this->windows) {
        std::string window_name = w.first;
        WindowConfiguration window_config = w.second;
        this->profiles[profile_name][window_name]["show"]                               = window_config.show;
        this->profiles[profile_name][window_name]["flags"]                              = (int)(window_config.flags);
        this->profiles[profile_name][window_name]["callback"]                           = window_config.callback;
        this->profiles[profile_name][window_name]["hotkey"]                             = { (int)(window_config.hotkey.Key()), window_config.hotkey.Modifiers().toInt() };
        this->profiles[profile_name][window_name]["position"]                           = { window_config.position.x, window_config.position.y };
        this->profiles[profile_name][window_name]["size"]                               = { window_config.size.x, window_config.size.y };
        this->profiles[profile_name][window_name]["soft_reset"]                         = window_config.soft_reset;
        this->profiles[profile_name][window_name]["reset_size"]                         = { window_config.reset_size.x, window_config.reset_size.y };

        this->profiles[profile_name][window_name]["param_config"]["show_hotkeys"]       = window_config.param_config.show_hotkeys;
        this->profiles[profile_name][window_name]["param_config"]["modules_list"]       = window_config.param_config.modules_list;
        this->profiles[profile_name][window_name]["param_config"]["module_filter"]      = window_config.param_config.module_filter;
        
        this->profiles[profile_name][window_name]["fpsms_config"]["show_options"]       = window_config.fpsms_config.show_options;
        this->profiles[profile_name][window_name]["fpsms_config"]["max_value_count"]    = window_config.fpsms_config.max_value_count;
        this->profiles[profile_name][window_name]["fpsms_config"]["max_delay"]          = window_config.fpsms_config.max_delay;
        this->profiles[profile_name][window_name]["fpsms_config"]["mode"]               = (int)window_config.fpsms_config.mode;

        this->profiles[profile_name][window_name]["font_config"]["font_name"]           = window_config.font_config.font_name;

        ///this->profiles[profile_name][window_name]["profile_reset"] = window_config.profile_reset; // Always true for a loaded profile
    }

    // Saving changes immediately to profile file.
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

    std::ofstream profilefile;
    profilefile.open(this->filename);

    /// Existing file with same name will be overwritten ...
    if (profilefile.is_open() && profilefile.good()) {
        profilefile << this->profiles.dump(4);
        profilefile.close();
        return true;
    }

    vislib::sys::Log::DefaultLog.WriteWarn(
        "[GUIWindowManager] Couldn't write to profile file: '%s'", this->filename.c_str());
    return false;
}


/**
 * GUIWindowManager::loadWindowConfigurationFile
 */
bool GUIWindowManager::loadWindowConfigurationFile(void) {

    std::ifstream profilefile;
    profilefile.open(this->filename);

    std::string line;
    std::stringstream stream;

    if (profilefile.is_open() && profilefile.good()) {
        while (std::getline(profilefile, line))
        {
            stream << line << std::endl;
        }
        profilefile.close();

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
        "[GUIWindowManager] Couldn't read profile file: '%s'", this->filename.c_str());
    return false;
}

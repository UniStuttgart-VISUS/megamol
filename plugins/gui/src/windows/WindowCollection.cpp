/*
 * WindowCollection.cpp
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */


#include "WindowCollection.h"
#include "Configurator.h"
#include "TransferFunctionEditor.h"
#include "LogConsole.h"
#include "PerformanceMonitor.h"
#include "ParameterList.h"


using namespace megamol;
using namespace megamol::gui;


WindowCollection::WindowCollection() {

    auto win_configurator = std::make_shared<Configurator>("Configurator");
    auto win_logconsole = std::make_shared<LogConsole>("Log Console");
    auto win_paramlist = std::make_shared<ParameterList>("Parameters");
    auto win_tfeditor = std::make_shared<TransferFunctionEditor>("Transfer Function Editor", false);
    auto win_perfmonitor = std::make_shared<PerformanceMonitor>("Performance Metrics");

    // Windows are sorted depending on hotkey
    this->windows.emplace_back(win_configurator);
    this->windows.emplace_back(win_paramlist);
    this->windows.emplace_back(win_logconsole);
    this->windows.emplace_back(win_tfeditor);
    this->windows.emplace_back(win_perfmonitor);

    win_configurator->SetData(win_tfeditor);
    win_paramlist->SetData(win_configurator, win_tfeditor, [&](const std::string &window_name) {
        this->AddWindow<ParameterList>(window_name);
    });
}


bool WindowCollection::AddWindow(const std::string &window_name, const std::function<void(WindowConfiguration::BasicConfig &)> &callback) {

    if (window_name.empty()) {
        megamol::core::utility::log::Log::DefaultLog.WriteWarn(
            "[GUI] Invalid window name. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
        return false;
    }
    auto win_hash = std::hash<std::string>()(window_name);
    if (this->WindowExists(win_hash)) {
        // Overwrite volatile callback for existing window
        for (auto& win : this->windows) {
            if (win->Hash() == win_hash) {
                win->SetVolatileCallback(callback);
                continue;
            }
        }
    }
    else {
        this->windows.push_back(std::make_shared<WindowConfiguration>(window_name, callback)); /// const_cast<std::function<void(WindowConfiguration::BasicConfig &)> &>(callback)
    }
    return true;
}


void WindowCollection::Update() {

    for (auto& win : this->windows) {
        win->Update();
    }
}


void WindowCollection::Draw(bool menu_visible) {
    
    const auto func = [&, this](WindowConfiguration& wc) {

        if (wc.Config().show) {
            ImGui::SetNextWindowBgAlpha(1.0f);
            ImGui::SetNextWindowCollapsed(wc.Config().collapsed, ImGuiCond_Always);

            // Begin Window
            if (!ImGui::Begin(wc.FullWindowTitle().c_str(), &wc.Config().show, wc.Config().flags)) {
                wc.Config().collapsed = ImGui::IsWindowCollapsed();
                ImGui::End(); // early ending
                return;
            }

            // Context menu of window
            bool collapsing_changed = false;
            wc.WindowContextMenu(menu_visible, collapsing_changed);

            // Draw window content
            wc.Draw();

            // Omit updating size and position of window from imgui for current frame when reset
            bool update_window_pos_size = !wc.Config().reset_pos_size;
            if (update_window_pos_size) {
                wc.Config().position = ImGui::GetWindowPos();
                wc.Config().size = ImGui::GetWindowSize();

                if (!collapsing_changed) {
                    wc.Config().collapsed = ImGui::IsWindowCollapsed();
                }
            }

            ImGui::End();
        }
    };

    this->EnumWindows(func);
}


void WindowCollection::PopUps() {

    for (auto& win : this->windows) {
        win->PopUps();
    }
}


bool WindowCollection::StateFromJSON(const nlohmann::json& in_json) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // First, search for not predefined window configurations and create additional windows
        for (auto &header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
                for (auto &config_item : header_item.value().items()) {
                    auto window_name = config_item.key();
                    auto win_hash = std::hash<std::string>()(window_name);
                    if (!this->WindowExists(win_hash)) {

                        int tmp_win_config_id = 0;
                        megamol::core::utility::get_json_value<int>(config_item.value(), {"win_callback"}, /// TODO rename to "win_config_id"
                            &tmp_win_config_id);
                        auto win_config_id = static_cast<WindowConfiguration::WindowConfigID>(tmp_win_config_id);

                        if (win_config_id == WindowConfiguration::WINDOW_ID_VOLATILE) {
                            this->AddWindow(window_name, std::function<void(WindowConfiguration::BasicConfig &)>());
                        } else if (win_config_id == WindowConfiguration::WINDOW_ID_PARAMETERS) {
                            this->AddWindow<ParameterList>(window_name);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                    "[GUI] Only additional volatile and custom parameter windows can be loaded from state file. [%s, %s, line %d]\n",
                                    __FILE__, __FUNCTION__, __LINE__);
                        }
                    }
                }
            }
        }

        // Then read configuration for all existing windows
        for (auto& window : this->windows) {
            window->StateFromJSON(in_json);
            window->SpecificStateFromJSON(in_json);
        }

#ifdef GUI_VERBOSE
        megamol::core::utility::log::Log::DefaultLog.WriteInfo("[GUI] Read window configurations from JSON string.");
#endif // GUI_VERBOSE

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

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window->Name()]["win_callback"] = static_cast<int>(window->WindowID()); /// TODO rename to "win_config_id"

            window->StateToJSON(inout_json);
            window->SpecificStateToJSON(inout_json);
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


bool WindowCollection::DeleteWindow(size_t win_hash_id) {

    for (auto iter = this->windows.begin(); iter != this->windows.end(); iter++) {
        if (((*iter)->Hash() == win_hash_id)) {
            if (((*iter)->WindowID() == WindowConfiguration::WINDOW_ID_VOLATILE) || ((*iter)->WindowID() == WindowConfiguration::WINDOW_ID_PARAMETERS)) {
                this->windows.erase(iter);
                return true;
            }
            else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                        "[GUI] Only volatile and custom parameter windows can be deleted. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
                return false;
            }
        }
    }
    return false;
}

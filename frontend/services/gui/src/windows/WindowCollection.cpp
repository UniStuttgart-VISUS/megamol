/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "WindowCollection.h"
#include "AnimationEditor.h"
#include "Configurator.h"
#include "HotkeyEditor.h"
#include "LogConsole.h"
#include "ParameterList.h"
#include "PerformanceMonitor.h"
#include "TransferFunctionEditor.h"


using namespace megamol;
using namespace megamol::gui;


WindowCollection::WindowCollection() : windows() {

    this->windows.emplace_back(std::make_shared<AnimationEditor>("Animation Editor"));
    this->windows.emplace_back(std::make_shared<HotkeyEditor>("Hotkey Editor"));
    this->windows.emplace_back(std::make_shared<LogConsole>("Log Console"));
    this->windows.emplace_back(std::make_shared<TransferFunctionEditor>("Transfer Function Editor", true));
    this->windows.emplace_back(std::make_shared<PerformanceMonitor>("Performance Metrics"));
    this->windows.emplace_back(
        std::make_shared<Configurator>("Configurator", this->GetWindow<TransferFunctionEditor>()));
    // Requires Configurator and TFEditor to be added before
    this->add_parameter_window("Parameters", AbstractWindow::WINDOW_ID_MAIN_PARAMETERS);

    // Windows are sorted depending on hotkey
    std::sort(this->windows.begin(), this->windows.end(),
        [&](std::shared_ptr<AbstractWindow> const& a, std::shared_ptr<AbstractWindow> const& b) {
            return (a->Config().hotkey.key > b->Config().hotkey.key);
        });

    // retrieve resource requests of each window class
    for (auto const& win : windows) {
        auto res = win->requested_lifetime_resources();
        requested_resources.insert(requested_resources.end(), res.begin(), res.end());
    }
    requested_resources.erase(
        std::unique(requested_resources.begin(), requested_resources.end()), requested_resources.end());
}


bool WindowCollection::AddWindow(
    const std::string& window_name, const std::function<void(AbstractWindow::BasicConfig&)>& callback) {

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
    } else {
        this->windows.push_back(std::make_shared<AbstractWindow>(
            window_name, const_cast<std::function<void(AbstractWindow::BasicConfig&)>&>(callback)));
    }
    return true;
}


void WindowCollection::Update() {

    // Call window update functions
    for (auto& win : this->windows) {
        win->Update();
    }
}


void WindowCollection::Draw(bool menu_visible) {

    const auto func = [&](AbstractWindow& wc) {
        if (wc.Config().show) {

            // Draw Window ----------------------------------------------------
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
            /// ImGui::PushID(static_cast<int>(wc.Hash()));
            wc.Draw();
            /// ImGui::PopID();

            // Reset or store window position and size
            if (wc.Config().reset_pos_size || (menu_visible && ImGui::IsMouseReleased(ImGuiMouseButton_Left) &&
                                                  ImGui::IsWindowFocused(ImGuiFocusedFlags_ChildWindows))) {
                wc.ApplyWindowSizePosition(menu_visible);
                wc.Config().reset_pos_size = false;
            } else {
                wc.Config().position = ImGui::GetWindowPos();
                wc.Config().size = ImGui::GetWindowSize();
                if (!collapsing_changed) {
                    wc.Config().collapsed = ImGui::IsWindowCollapsed();
                }
            }

            ImGui::End();

            // Draw Pop-ups ---------------------------------------------------
            wc.PopUps();
        }
    };

    this->EnumWindows(func);
}


bool WindowCollection::StateFromJSON(const nlohmann::json& in_json) {

    try {
        if (!in_json.is_object()) {
            megamol::core::utility::log::Log::DefaultLog.WriteError(
                "[GUI] Invalid JSON object. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }

        // First, search for not predefined window configurations and create additional windows
        for (auto& header_item : in_json.items()) {
            if (header_item.key() == GUI_JSON_TAG_WINDOW_CONFIGS) {
                for (auto& config_item : header_item.value().items()) {
                    auto window_name = config_item.key();
                    auto win_hash = std::hash<std::string>()(window_name);
                    if (!this->WindowExists(win_hash)) {

                        int tmp_win_config_id = 0;
                        megamol::core::utility::get_json_value<int>(config_item.value(),
                            {"win_callback"}, /// XXX rename to "win_config_id"
                            &tmp_win_config_id);
                        auto win_config_id = static_cast<AbstractWindow::WindowConfigID>(tmp_win_config_id);

                        if (win_config_id == AbstractWindow::WINDOW_ID_VOLATILE) {
                            this->AddWindow(window_name, std::function<void(AbstractWindow::BasicConfig&)>());
                        } else if (win_config_id == AbstractWindow::WINDOW_ID_PARAMETERS) {
                            this->add_parameter_window(window_name, AbstractWindow::WINDOW_ID_PARAMETERS);
                        } else {
                            megamol::core::utility::log::Log::DefaultLog.WriteError(
                                "[GUI] Only additional volatile and custom parameter windows can be loaded from state "
                                "file. [%s, %s, line %d]\n",
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

            inout_json[GUI_JSON_TAG_WINDOW_CONFIGS][window->Name()]["win_callback"] =
                static_cast<int>(window->WindowID()); /// XXX rename to "win_config_id"

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
            if (((*iter)->WindowID() == AbstractWindow::WINDOW_ID_VOLATILE) ||
                ((*iter)->WindowID() == AbstractWindow::WINDOW_ID_PARAMETERS)) {
                this->windows.erase(iter);
                return true;
            } else {
                megamol::core::utility::log::Log::DefaultLog.WriteError(
                    "[GUI] Only volatile and custom parameter windows can be deleted. [%s, %s, line %d]\n", __FILE__,
                    __FUNCTION__, __LINE__);
                return false;
            }
        }
    }
    return false;
}


void megamol::gui::WindowCollection::setRequestedResources(
    std::shared_ptr<frontend_resources::FrontendResourcesMap> const& resources) {
    for (auto& win : windows) {
        win->setRequestedResources(resources);
    }
}


void WindowCollection::add_parameter_window(
    const std::string& window_name, AbstractWindow::WindowConfigID win_id, ImGuiID initial_module_uid) {

    if ((win_id == AbstractWindow::WINDOW_ID_MAIN_PARAMETERS) || (win_id == AbstractWindow::WINDOW_ID_PARAMETERS)) {
        auto win_paramlist = std::make_shared<ParameterList>(window_name, win_id, initial_module_uid,
            this->GetWindow<Configurator>(), this->GetWindow<TransferFunctionEditor>(),
            [&](const std::string& windowname, AbstractWindow::WindowConfigID winid, ImGuiID initialmoduleuid) {
                this->add_parameter_window(windowname, winid, initialmoduleuid);
            });
        this->windows.emplace_back(win_paramlist);
    }
}

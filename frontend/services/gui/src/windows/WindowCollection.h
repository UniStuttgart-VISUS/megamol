/*
 * WindowCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once


#include "AbstractWindow.h"
#include "FrontendResource.h"
#include "imgui.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"
#include <functional>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>
#include <set>


namespace megamol::gui {

/** ************************************************************************
 * This class hold the GUI windows and controls the placement and appearance
 */
class WindowCollection {
public:
    WindowCollection();
    ~WindowCollection() = default;

    void Update();
    void Draw(bool menu_visible);

    bool StateFromJSON(const nlohmann::json& in_json);
    bool StateToJSON(nlohmann::json& inout_json);

    bool AddWindow(const std::string& window_name, const std::function<void(AbstractWindow::BasicConfig&)>& callback);

    template<typename T, typename... Args>
    bool AddWindow(const std::string& window_name, Args... args) {
        if (window_name.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                "[GUI] Invalid window name. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        this->created_windows[window_name] = std::make_shared<T>(window_name, std::forward<Args>(args)...);
        return true;
    }

    //inline void EnumWindows(const std::function<void(AbstractWindow&)>& cb) {
    //    // Needs fixed size if window is added while looping
    //    for (auto& [key, val] : this->windows) {
    //        cb((*val));
    //    }
    //}

    void EnumAvailWindows(const std::function<void(AbstractWindow&)>& cb) {
        for (auto& win : avail_windows) {
            cb(*win);
        }
    }

    void EnumCreatedWindows(const std::function<void(AbstractWindow&)>& cb) {
        for (auto& [key, val] : this->created_windows) {
            cb((*val));
        }
    }

    inline bool WindowExists(size_t hash_id) const {
        for (auto& [key, val] : this->created_windows) {
            if (val->Hash() == hash_id)
                return true;
        }
        return false;
    }

    template<typename T>
    std::shared_ptr<T> GetWindow() const {
        for (auto& [key, win_ptr] : this->created_windows) {
            if (auto ret_win_ptr = std::dynamic_pointer_cast<T>(win_ptr))
                return ret_win_ptr;
        }
        return nullptr;
    }

    bool DeleteWindow(size_t hash_id);

    std::vector<std::string> requested_lifetime_resources() const {
        return requested_resources;
    }

    void setRequestedResources(std::shared_ptr<frontend_resources::FrontendResourcesMap> const& resources);

    void digestChangedRequestedResources();

private:
    // VARIABLES ------------------------------------------------------

    std::set<frontend_resources::KeyCode, std::string> registered_windows;

    std::vector<std::shared_ptr<AbstractWindow>> avail_windows;

    std::unordered_map<std::string, std::shared_ptr<AbstractWindow>> created_windows;

    std::vector<std::string> requested_resources;

    // FUNCTIONS ------------------------------------------------------

    void add_parameter_window(const std::string& window_name, AbstractWindow::WindowConfigID win_id,
        ImGuiID initial_module_uid = GUI_INVALID_ID);
};

} // namespace megamol::gui

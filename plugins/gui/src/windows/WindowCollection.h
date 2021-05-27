/*
 * WindowCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED
#define MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED
#pragma once


#include <functional>
#include <map>
#include <string>
#include <vector>
#include "imgui.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"
#include "WindowConfiguration.h"


namespace megamol {
namespace gui {

    /** ************************************************************************
     * This class hold the GUI windows and controls the placement and appearance
     */
    class WindowCollection {
    public:

        WindowCollection();
        ~WindowCollection() = default;

        void Update();
        void Draw(bool menu_visible);
        void PopUps();

        bool StateFromJSON(const nlohmann::json& in_json);
        bool StateToJSON(nlohmann::json& inout_json);

        bool AddWindow(const std::string& window_name, const std::function<void(WindowConfiguration::BasicConfig&)>& callback);

        template<typename T>
        bool AddWindow(const std::string &window_name);

        inline void EnumWindows(const std::function<void(WindowConfiguration&)>& cb) {
            // Needs fixed size if window is added while looping
            auto window_count = this->windows.size();
            for (size_t i = 0; i < window_count; i++) {
                cb((*this->windows[i]));
            }
        }

        inline bool WindowExists(size_t hash_id) const {
            for (auto& wc : this->windows) {
                if (wc->Hash() == hash_id)
                    return true;
            }
            return false;
        }

        template<typename T>
        std::shared_ptr<T> GetWindow() const {
            for (auto& win_ptr : this->windows) {
                if (auto ret_win_ptr = std::dynamic_pointer_cast<T>(win_ptr))
                    return ret_win_ptr;
            }
            return nullptr;
        }

        bool DeleteWindow(size_t hash_id);

    private:

        // VARIABLES ------------------------------------------------------

        std::vector<std::shared_ptr<WindowConfiguration>> windows;
    };

    template<typename T>
    bool WindowCollection::AddWindow(const std::string &window_name) {

        if (window_name.empty()) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[GUI] Invalid window name. [%s, %s, line %d]\n", __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        auto win_hash = std::hash<std::string>()(window_name);
        if (this->WindowExists(win_hash)) {
            megamol::core::utility::log::Log::DefaultLog.WriteWarn(
                    "[GUI] Found already existing window with name '%s'. Window names must be unique. [%s, %s, line %d]\n",
                    window_name.c_str(), __FILE__, __FUNCTION__, __LINE__);
            return false;
        }
        this->windows.push_back(std::make_shared<T>(window_name));
        return true;
    }

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED

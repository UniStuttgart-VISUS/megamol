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
#include "AbstractWindow.h"
#include "imgui.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"


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

        bool AddWindow(
            const std::string& window_name, const std::function<void(AbstractWindow::BasicConfig&)>& callback);

        inline void EnumWindows(const std::function<void(AbstractWindow&)>& cb) {
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

        std::vector<std::shared_ptr<AbstractWindow>> windows;

        // FUNCTIONS ------------------------------------------------------

        void add_parameter_window(const std::string& window_name, AbstractWindow::WindowConfigID win_id,
            ImGuiID initial_module_uid = GUI_INVALID_ID);
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED

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

    /**
     * This class controls the placement and appearance of windows.
     */
    class WindowCollection {
    public:

        WindowCollection();

        ~WindowCollection() = default;

        inline void Update() {
            for (auto& win : this->windows) {
                win->Update();
            }
        }

        inline void Draw() {
            for (auto& win : this->windows) {
                win->Draw();
            }
        }

        inline void PopUps() {
            for (auto& win : this->windows) {
                win->PopUps();
            }
        }

        bool AddWindow(WindowConfiguration& wc);

        inline void EnumWindows(std::function<void(WindowConfiguration&)> cb) {
            // Needs fixed size if window is added while looping
            auto window_count = this->windows.size();
            for (size_t i = 0; i < window_count; i++) {
                cb((*this->windows[i]));
            }
        }

        inline WindowConfiguration& GetWindow(WindowConfiguration::WindowConfigID win_id) {

        }

        bool DeleteWindow(size_t hash_id);

        inline bool WindowExists(size_t hash_id) const {
            for (auto& wc : this->windows) {
                if (wc->Hash() == hash_id)
                    return true;
            }
            return false;
        }

        // --------------------------------------------------------------------
        // STATE

        /**
         * Deserializes a window configuration state.
         * Should be called before(!) ImGui::Begin() because existing window configurations are overwritten.
         *
         * @param json  The JSON to deserialize from.
         *
         * @return True on success, false otherwise.
         */
        bool StateFromJSON(const nlohmann::json& in_json);

        /**
         * Serializes the current window configuration.
         *
         * @param json  The JSON to serialize to.
         *
         * @return True on success, false otherwise.
         */
        bool StateToJSON(nlohmann::json& inout_json);

    private:

        // VARIABLES ------------------------------------------------------

        std::vector<std::unique_ptr<WindowConfiguration>> windows;
    };

} // namespace gui
} // namespace megamol

#endif // MEGAMOL_GUI_WINDOWCOLLECTION_INCLUDED

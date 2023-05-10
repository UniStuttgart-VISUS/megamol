/*
 * WindowCollection.h
 *
 * Copyright (C) 2019 by Universitaet Stuttgart (VIS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "FrontendResource.h"
#include "FrontendResourcesMap.h"
#include "gui_utils.h"
#include "mmcore/utility/JSONHelper.h"
#include "mmcore/utility/log/Log.h"
#include "mmcore/view/Input.h"


namespace megamol::gui {

/** ************************************************************************
 * This class holds the configuration of a GUI window
 */
class AbstractWindow {
public:
    /** Identifiers for the predefined window draw callbacks. */
    /// XXX Keep explicit numbers for backward compatibility
    enum WindowConfigID {
        WINDOW_ID_VOLATILE = 0,
        WINDOW_ID_MAIN_PARAMETERS = 1,
        WINDOW_ID_PARAMETERS = 2,
        WINDOW_ID_PERFORMANCE = 3,
        WINDOW_ID_HOTKEYEDITOR = 4,
        WINDOW_ID_TRANSFER_FUNCTION = 5,
        WINDOW_ID_CONFIGURATOR = 6,
        WINDOW_ID_LOGCONSOLE = 7,
        WINDOW_ID_ANIMATIONEDITOR = 8
    };

    struct BasicConfig {
        bool show = false;                          // [SAVED] show/hide window
        ImGuiWindowFlags flags = 0;                 // [SAVED] imgui window flags
        megamol::core::view::KeyCode hotkey;        // [SAVED] hotkey for opening/closing window
        ImVec2 position = ImVec2(0.0f, 0.0f);       // [SAVED] position for reset on state loading (current position)
        ImVec2 size = ImVec2(0.0f, 0.0f);           // [SAVED] size for reset on state loading (current size)
        ImVec2 reset_size = ImVec2(0.0f, 0.0f);     // [SAVED] minimum window size for soft reset
        ImVec2 reset_position = ImVec2(0.0f, 0.0f); // [SAVED] window position for minimize reset
        bool collapsed = false;                     // [SAVED] flag indicating whether window is collapsed or not
        bool reset_pos_size = true;                 // flag indicates whether to reset window position and size
    };

    typedef std::function<void(AbstractWindow::BasicConfig&)> VolatileDrawCallback_t;

    virtual std::vector<std::string> requested_lifetime_resources() const {
        return std::vector<std::string>();
    }

    virtual void setRequestedResources(std::shared_ptr<frontend_resources::FrontendResourcesMap> const& resources) {
        frontend_resources = resources;
    };

    AbstractWindow(const std::string& name, WindowConfigID window_id)
            : win_config()
            , win_hotkeys()
            , hash_id(std::hash<std::string>()(name))
            , name(name)
            , window_id(window_id)
            , volatile_draw_callback(nullptr) {}

    AbstractWindow(const std::string& name, VolatileDrawCallback_t& callback)
            : win_config()
            , win_hotkeys()
            , hash_id(std::hash<std::string>()(name))
            , name(name)
            , window_id(WINDOW_ID_VOLATILE)
            , volatile_draw_callback(callback) {}

    ~AbstractWindow() = default;

    std::string Name() const {
        return this->name;
    }

    size_t Hash() const {
        return this->hash_id;
    }

    inline WindowConfigID WindowID() const {
        return this->window_id;
    }

    inline BasicConfig& Config() {
        return this->win_config;
    }

    inline std::string FullWindowTitle() const {
        return (this->Name() + "     " + this->win_config.hotkey.ToString());
    }

    void SetVolatileCallback(std::function<void(AbstractWindow::BasicConfig&)> const& callback) {
        this->volatile_draw_callback = const_cast<std::function<void(AbstractWindow::BasicConfig&)>&>(callback);
        this->window_id = WINDOW_ID_VOLATILE;
    }

    void ApplyWindowSizePosition(bool consider_menu);

    void WindowContextMenu(bool menu_visible, bool& out_collapsing_changed);

    void StateFromJSON(const nlohmann::json& in_json);

    void StateToJSON(nlohmann::json& inout_json);

    inline megamol::gui::HotkeyMap_t& GetHotkeys() {
        return this->win_hotkeys;
    }

    // --------------------------------------------------------------------
    // IMPLEMENT

    virtual bool Update() {
        return true;
    }

    virtual bool Draw() {
        if ((window_id == WINDOW_ID_VOLATILE) && (volatile_draw_callback != nullptr)) {
            volatile_draw_callback(this->win_config);
            return true;
        }
        return false;
    }

    virtual void PopUps() {}

    virtual void SpecificStateToJSON(nlohmann::json& inout_json) {}

    virtual void SpecificStateFromJSON(const nlohmann::json& in_json){};

protected:
    BasicConfig win_config;
    megamol::gui::HotkeyMap_t win_hotkeys;
    WindowConfigID window_id; // [SAVED] ID of the predefined callback drawing the window content
    std::shared_ptr<frontend_resources::FrontendResourcesMap> frontend_resources;

private:
    size_t hash_id;   // unique hash generated from name to omit string comparison
    std::string name; // [SAVED] unique name of the window
    VolatileDrawCallback_t volatile_draw_callback;
};

} // namespace megamol::gui

#pragma once

#include "KeyboardMouseInput.h"
#include "WindowConfig.h"
#include "gui_utils.h"

#include "FrontendResourcesMap.h"
#include "nlohmann/json.hpp"

namespace megamol::gui {

class ResourceInterface {
public:
    virtual std::vector<std::string> requested_lifetime_resources() const {
        return std::vector<std::string>();
    }

    virtual void setRequestedResources(std::shared_ptr<frontend_resources::FrontendResourcesMap> const& resources) {
        frontend_resources = resources;
    };

    virtual void digestChangedRequestedResources() {}

private:
    std::shared_ptr<frontend_resources::FrontendResourcesMap> frontend_resources;
};

class JSONSerializable {
public:
    virtual void StateFromJSON(const nlohmann::json& in_json) = 0;

    virtual void StateToJSON(nlohmann::json& inout_json) const = 0;
};

class AbstractWindow2 : public ResourceInterface, JSONSerializable {
public:
    AbstractWindow2(std::string const& name);

    std::string Name() const {
        return name_;
    }

    BasicConfig& Config() {
        return this->win_config_;
    }

    std::string FullWindowTitle() const {
        return (this->Name() + "     " + this->win_config_.hotkey.ToString());
    }

    megamol::gui::HotkeyMap_t& GetHotkeys() {
        return this->win_hotkeys_;
    }

    void ApplyWindowSizePosition(bool consider_menu);

    void WindowContextMenu(bool menu_visible, bool& out_collapsing_changed);

    void StateFromJSON(const nlohmann::json& in_json) override;

    void StateToJSON(nlohmann::json& inout_json) const override;

    // --------------------------------------------------------------------
    // IMPLEMENT

    virtual bool Update() {
        return true;
    }

    virtual bool Draw() {
        /*if ((window_id == WINDOW_ID_VOLATILE) && (volatile_draw_callback != nullptr)) {
            volatile_draw_callback(this->win_config);
            return true;
        }*/
        return true;
    }

    virtual void PopUps() {}

    virtual void SpecificStateToJSON(nlohmann::json& inout_json) {}

    virtual void SpecificStateFromJSON(const nlohmann::json& in_json){};

protected:
    BasicConfig win_config_;
    megamol::gui::HotkeyMap_t win_hotkeys_;

private:
    std::string name_;
};
} // namespace megamol::gui

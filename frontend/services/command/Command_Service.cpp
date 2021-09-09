#include "Command_Service.hpp"

#include "KeyboardMouse_Events.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/utility/log/Log.h"


static void log(std::string const& text) {
    const std::string msg = "Command_Service: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteInfo(msg.c_str());
}


static void log_error(std::string const& text) {
    const std::string msg = "Command_Service: " + text;
    megamol::core::utility::log::Log::DefaultLog.WriteError(msg.c_str());
}


bool megamol::frontend::Command_Service::init(void* configPtr) {

    requestedResourceNames = {"KeyboardEvents"};
    providedResourceReferences = {{frontend_resources::CommandRegistry_Req_Name, commands}};

    log("initialized successfully");

    return true;
}


void megamol::frontend::Command_Service::close() {

}

void megamol::frontend::Command_Service::digestChangedRequestedResources() {
        auto keyboard_events =
            &this->requestedResourceReferences[0].getResource<megamol::frontend_resources::KeyboardEvents>();
        for (auto& key_event : keyboard_events->key_events) {
            auto key = std::get<0>(key_event);
            auto action = std::get<1>(key_event);
            auto modifiers = std::get<2>(key_event);

            if (key == frontend_resources::Key::KEY_RIGHT_SHIFT || key == frontend_resources::Key::KEY_LEFT_SHIFT ||
                key == frontend_resources::Key::KEY_RIGHT_CONTROL|| key == frontend_resources::Key::KEY_LEFT_CONTROL||
                key == frontend_resources::Key::KEY_RIGHT_ALT || key == frontend_resources::Key::KEY_LEFT_ALT) {
                commands.modifiers_changed(modifiers);
            }

            if (action == frontend_resources::KeyAction::PRESS) {
                frontend_resources::KeyCode kc {key, modifiers};
                auto p = commands.param_from_keycode(kc);
                if (p != nullptr) {
                    p->setDirty();
                }
            }
        }
}

void megamol::frontend::Command_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    requestedResourceReferences = resources;
}


/**
 * MegaMol
 * Copyright (c) 2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "Command_Service.hpp"

#include "KeyboardMouse_Events.h"
#include "ModuleGraphSubscription.h"
#include "mmcore/MegaMolGraph.h"
#include "mmcore/param/AbstractParam.h"
#include "mmcore/param/ButtonParam.h"
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

    requestedResourceNames = {
        "optional<KeyboardEvents>",
        frontend_resources::MegaMolGraph_Req_Name,
        frontend_resources::MegaMolGraph_SubscriptionRegistry_Req_Name,
    };
    providedResourceReferences = {{frontend_resources::CommandRegistry_Req_Name, commands}};

    log("initialized successfully");

    return true;
}


void megamol::frontend::Command_Service::close() {}

void megamol::frontend::Command_Service::digestChangedRequestedResources() {
    auto maybe_keyboard_events =
        this->requestedResourceReferences[0].getOptionalResource<megamol::frontend_resources::KeyboardEvents>();

    if (!maybe_keyboard_events.has_value()) {
        return;
    }

    megamol::frontend_resources::KeyboardEvents const& keyboard_events = maybe_keyboard_events.value().get();

    for (auto& key_event : keyboard_events.key_events) {
        auto key = std::get<0>(key_event);
        auto action = std::get<1>(key_event);
        auto modifiers = std::get<2>(key_event);

        if (key == frontend_resources::Key::KEY_RIGHT_SHIFT || key == frontend_resources::Key::KEY_LEFT_SHIFT ||
            key == frontend_resources::Key::KEY_RIGHT_CONTROL || key == frontend_resources::Key::KEY_LEFT_CONTROL ||
            key == frontend_resources::Key::KEY_RIGHT_ALT || key == frontend_resources::Key::KEY_LEFT_ALT) {
            commands.modifiers_changed(modifiers);
        }

        if (action == frontend_resources::KeyAction::PRESS) {
            frontend_resources::KeyCode kc{key, modifiers};
            commands.exec_command(kc);
        }
    }
}

void megamol::frontend::Command_Service::setRequestedResources(std::vector<FrontendResource> resources) {
    requestedResourceReferences = resources;

    auto& megamolgraph = const_cast<megamol::core::MegaMolGraph&>(
        requestedResourceReferences[1].getResource<megamol::core::MegaMolGraph>());

    auto& megamolgraph_subscription = const_cast<frontend_resources::MegaMolGraph_SubscriptionRegistry&>(
        requestedResourceReferences[2].getResource<frontend_resources::MegaMolGraph_SubscriptionRegistry>());

    frontend_resources::ModuleGraphSubscription command_registry_subscription("Command Registry");

    // this lambda is replicated in HotkeyEditor.h for the GUI to use
    frontend_resources::Command::EffectFunction Parameter_Lambda = [&](const frontend_resources::Command* self) {
        auto my_p = megamolgraph.FindParameter(self->parent);
        if (my_p != nullptr) {
            my_p->setDirty();
        }
    };

    command_registry_subscription.AddModule = [&, Parameter_Lambda](core::ModuleInstance_t const& module_inst) {
        // iterate parameters, add hotkeys to CommandRegistry
        auto module_ptr = module_inst.modulePtr;
        for (auto child = module_ptr->ChildList_Begin(); child != module_ptr->ChildList_End(); ++child) {
            auto ps = dynamic_cast<core::param::ParamSlot*>((*child).get());
            if (ps != nullptr) {
                auto p = ps->Param<core::param::ButtonParam>();
                if (p != nullptr) {
                    frontend_resources::Command c;
                    c.key = p->GetKeyCode();
                    c.parent = ps->FullName();
                    c.name = module_ptr->Name().PeekBuffer() + std::string("_") + ps->Name().PeekBuffer();
                    c.effect = Parameter_Lambda;
                    this->commands.add_command(c);
                }
            }
        }
        return true;
    };

    command_registry_subscription.RenameModule = [&](std::string const& oldName, std::string const& newName,
                                                     core::ModuleInstance_t const& module_inst) {
        for (auto child = module_inst.modulePtr->ChildList_Begin(); child != module_inst.modulePtr->ChildList_End();
             ++child) {
            auto ps = dynamic_cast<core::param::ParamSlot*>((*child).get());
            if (ps != nullptr) {
                auto p = ps->Param<core::param::ButtonParam>();
                if (p != nullptr) {
                    auto command_name = oldName + std::string("_") + ps->Name().PeekBuffer();
                    auto updated_command_name = newName + std::string("_") + ps->Name().PeekBuffer();
                    auto c = this->commands.get_command(command_name);
                    this->commands.remove_command_by_name(command_name);
                    c.name = updated_command_name;
                    c.parent = ps->FullName();
                    this->commands.add_command(c);
                }
            }
        }
        return true;
    };

    command_registry_subscription.DeleteModule = [&](core::ModuleInstance_t const& module_inst) {
        // iterate parameters, remove hotkeys from CommandRegistry
        auto module_ptr = module_inst.modulePtr;
        for (auto child = module_ptr->ChildList_Begin(); child != module_ptr->ChildList_End(); ++child) {
            auto ps = dynamic_cast<core::param::ParamSlot*>((*child).get());
            if (ps != nullptr) {
                auto p = ps->Param<core::param::ButtonParam>();
                if (p != nullptr) {
                    this->commands.remove_command_by_parent(ps->FullName().PeekBuffer());
                }
            }
        }
        return true;
    };

    megamolgraph_subscription.subscribe(command_registry_subscription);
}

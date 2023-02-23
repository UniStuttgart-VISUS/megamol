/*
 * ModuleGraphSubscription.h
 *
 * Copyright (C) 2022 by VISUS (Universitaet Stuttgart).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include <functional>
#include <string>

#include <mmcore/MegaMolGraphTypes.h>
#include <mmcore/param/ParamSlot.h>

namespace megamol::frontend_resources {

static std::string MegaMolGraph_Req_Name = "MegaMolGraph";

static std::string MegaMolGraph_SubscriptionRegistry_Req_Name = "MegaMolGraph_SubscriptionRegistry";

/**
 * Interested parties may subscribe to changes of the MegaMolGraph using this subscription API.
 * Changes to the grpah will be passed to the subscribers using the callbacks provided by the subscriber.
 * Each subscriber needs a name which will be used for error messages.
 */
struct ModuleGraphSubscription {
private:
    std::string name;

public:
    explicit ModuleGraphSubscription(std::string const& subscriber_name) : name{subscriber_name} {}

    std::string const& Name() const {
        return name;
    }

    /**
     * Informs subscriber about a new module which has been added to the graph.
     * Module already exists and has been created successfully before this callback is called.
     */
    std::function<bool(core::ModuleInstance_t const&)> AddModule = [](auto const&) { return true; };

    /**
     * Informs subscriber about deletion of a module.
     * Module still exists but will be deleted soon after this callback finished.
     */
    std::function<bool(core::ModuleInstance_t const&)> DeleteModule = [](auto const&) { return true; };

    using ParamSlotPtr = core::param::ParamSlot*;

    /**
     * Informs subscriber about newly created parameters of module(s)
     * The parameters and the respective module have been created successfully when this callback is called.
     * This callback gets called after the AddModule callback.
     * All ParamSlotPtr values are non-null.
     */
    std::function<bool(std::vector<ParamSlotPtr> const& /*params*/)> AddParameters = [](auto const&) { return true; };

    /**
     * Tells subscriber about deleted parameters (due to module deletion)
     * The parameters still exist when the callback is executed, but get deleted soon after the callback finished.
     * This callback gets called before the DeleteModule callback.
     * All ParamSlotPtr values are non-null.
     */
    std::function<bool(std::vector<ParamSlotPtr> const& /*params*/)> RemoveParameters = [](auto const&) {
        return true;
    };

    /**
     * Notifies subscriber about change of a paramter value, providing the parameter and its new value.
     * Gets called after parameter value changed to new value.
     * All ParamSlotPtr values are non-null.
     */
    std::function<bool(ParamSlotPtr const& /*param*/, std::string const& /*new_value*/)> ParameterChanged =
        [](auto const&, auto const&) { return true; };

    /**
     * Informs about renaming of a module.
     * The module has already been renamed successfully when the callback is executed.
     */
    std::function<bool(std::string const& /*old name*/, std::string const& /*new name*/, core::ModuleInstance_t const&)>
        RenameModule = [](auto const&, auto const&, auto const&) { return true; };

    /**
     * Informs about successfull creation of a new call.
     * Call already exists when the callback is called.
     */
    std::function<bool(core::CallInstance_t const&)> AddCall = [](auto const&) { return true; };

    /**
     * Informs about deletion of a call.
     * The call still exists and will be deleted soon after execution of this callback.
     */
    std::function<bool(core::CallInstance_t const&)> DeleteCall = [](auto const&) { return true; };

    /**
     * Informs about a graph module (view) becoming a graph entry point, a graph module that is poked for rendering
     */
    std::function<bool(core::ModuleInstance_t const&)> EnableEntryPoint = [](auto const&) { return true; };

    /**
     * Informs about a graph module (view) becoming disabled as an entry point
     */
    std::function<bool(core::ModuleInstance_t const&)> DisableEntryPoint = [](auto const&) { return true; };
};

struct MegaMolGraph_SubscriptionRegistry {
    // only use these public API functions to de/register subscribers
    void subscribe(ModuleGraphSubscription subscriber);

    // only use these public API functions to de/register subscribers
    void unsubscribe(std::string const& subscriber_name);

    // do not touch.
    // the subscribers are accessed by the graph itself and should not be touched by outsiders
    std::vector<ModuleGraphSubscription> subscribers;

    std::pair<bool, std::string> tell_all(std::function<bool(ModuleGraphSubscription& subscriber)> callback) {
        bool result = true;
        std::string name;
        for (auto& s : subscribers) {
            if (!callback(s)) {
                result = false;
                name = s.Name();
            }
        }
        return std::pair{result, name};
    }
};

} // namespace megamol::frontend_resources

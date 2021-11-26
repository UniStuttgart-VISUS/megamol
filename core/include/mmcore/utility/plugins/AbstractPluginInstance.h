/**
 * MegaMol
 * Copyright (c) 2015-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_UTILITY_PLUGINS_ABSTRACTPLUGININSTANCE_H_INCLUDED
#define MEGAMOLCORE_UTILITY_PLUGINS_ABSTRACTPLUGININSTANCE_H_INCLUDED
#pragma once

#include <memory>
#include <string>

#include "mmcore/factories/AbstractObjectFactoryInstance.h"

namespace megamol::core::utility::plugins {

/**
 * Abstract base class for all object descriptions.
 *
 * An object is described using a unique name. This name is compared case
 * insensitive!
 */
class AbstractPluginInstance : public factories::AbstractObjectFactoryInstance {
public:
    /** The shared pointer type to be used */
    typedef std::shared_ptr<AbstractPluginInstance const> ptr_type;

    /** deleted copy ctor */
    AbstractPluginInstance(const AbstractPluginInstance& src) = delete;

    /** deleted assignment operatior */
    AbstractPluginInstance& operator=(const AbstractPluginInstance& rhs) = delete;

    /**
     * Answer the (machine-readable) name of the plugin.
     *
     * @return The (machine-readable) name of the plugin
     */
    const std::string& GetObjectFactoryName() const override {
        return name;
    }

    /**
     * Answer the (human-readable) description of the plugin.
     *
     * @return The (human-readable) description of the plugin.
     */
    virtual const std::string& GetDescription() const {
        return description;
    }

    /**
     * Answer the call description manager of the plugin.
     *
     * @return The call description manager of the plugin.
     */
    const factories::CallDescriptionManager& GetCallDescriptionManager() const override;

    /**
     * Answer the module description manager of the plugin.
     *
     * @return The module description manager of the plugin.
     */
    const factories::ModuleDescriptionManager& GetModuleDescriptionManager() const override;

protected:
    /** Ctor. */
    AbstractPluginInstance(const char* name, const char* description);

    /** Dtor. */
    ~AbstractPluginInstance() override;

    /**
     * This factory methode registers all module and call classes exported
     * by this plugin instance at the respective factories.
     *
     * @remarks This method is automatically called when the factories are
     *          accessed for the first time. Do not call manually.
     */
    virtual void registerClasses() = 0;

private:
    /** Ensures that registered classes was called */
    void ensureRegisterClassesWrapper() const;

    /** The (machine-readable) name of the plugin */
    std::string name;

    /** The (human-readable) description of the plugin */
    std::string description;

    /** Flag whether or not the module and call classes have been registered */
    bool classes_registered;
};

} // namespace megamol::core::utility::plugins

#endif // MEGAMOLCORE_UTILITY_PLUGINS_ABSTRACTPLUGININSTANCE_H_INCLUDED

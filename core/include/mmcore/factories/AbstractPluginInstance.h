/**
 * MegaMol
 * Copyright (c) 2015, MegaMol Dev Team
 * All rights reserved.
 */

#pragma once

#include <memory>
#include <string>

#include "CallDescriptionManager.h"
#include "ModuleDescriptionManager.h"

namespace megamol::core::factories {

/**
 * Abstract base class for all object descriptions.
 *
 * An object is described using a unique name. This name is compared case
 * insensitive!
 */
class AbstractPluginInstance {
public:
    /** The shared pointer type to be used */
    typedef std::shared_ptr<AbstractPluginInstance const> ptr_type;

    /** delete copy constructor */
    AbstractPluginInstance(const AbstractPluginInstance&) = delete;

    /** delete move constructor */
    AbstractPluginInstance(AbstractPluginInstance&&) = delete;

    /** delete copy assignment */
    AbstractPluginInstance& operator=(const AbstractPluginInstance&) = delete;

    /** delete move assignment */
    AbstractPluginInstance& operator=(AbstractPluginInstance&&) = delete;

    /**
     * Answer the (machine-readable) name of the plugin.
     *
     * @return The (machine-readable) name of the plugin
     */
    inline const std::string& GetObjectFactoryName() const {
        return name;
    }

    /**
     * Answer the (human-readable) description of the plugin.
     *
     * @return The (human-readable) description of the plugin.
     */
    inline const std::string& GetDescription() const {
        return description;
    }

    /**
     * Answer the call description manager of the plugin.
     *
     * @return The call description manager of the plugin.
     */
    const factories::CallDescriptionManager& GetCallDescriptionManager() const;

    /**
     * Answer the module description manager of the plugin.
     *
     * @return The module description manager of the plugin.
     */
    const factories::ModuleDescriptionManager& GetModuleDescriptionManager() const;

protected:
    /** Ctor. */
    AbstractPluginInstance(const char* name, const char* description);

    /** Dtor. */
    virtual ~AbstractPluginInstance();

    /**
     * This factory methode registers all module and call classes exported
     * by this plugin instance at the respective factories.
     *
     * @remarks This method is automatically called when the factories are
     *          accessed for the first time. Do not call manually.
     */
    virtual void registerClasses() = 0;

    /** The call description manager of the factory. */
    CallDescriptionManager call_descriptions;

    /** The module description manager of the factory. */
    ModuleDescriptionManager module_descriptions;

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

} // namespace megamol::core::factories

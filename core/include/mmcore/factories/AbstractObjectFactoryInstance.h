/**
 * MegaMol
 * Copyright (c) 2015-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_FACTORIES_ABSTRACTOBJECTFACTORYINSTANCE_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_ABSTRACTOBJECTFACTORYINSTANCE_H_INCLUDED
#pragma once

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
class AbstractObjectFactoryInstance {
public:
    /** delete copy constructor */
    AbstractObjectFactoryInstance(const AbstractObjectFactoryInstance& other) = delete;

    /** delete move constructor */
    AbstractObjectFactoryInstance(AbstractObjectFactoryInstance&& other) = delete;

    /** delete copy assignment */
    AbstractObjectFactoryInstance& operator=(const AbstractObjectFactoryInstance& other) = delete;

    /** delete move assignment */
    AbstractObjectFactoryInstance& operator=(AbstractObjectFactoryInstance&& other) = delete;

    /**
     * Answer the (machine-readable) name of the factory. This usually is
     * the name of the plugin.
     *
     * @return The (machine-readable) name of the factory.
     */
    virtual const std::string& GetObjectFactoryName() const = 0;

    /**
     * Answer the call description manager of the factory.
     *
     * @return The call description manager of the factory.
     */
    virtual const CallDescriptionManager& GetCallDescriptionManager() const {
        return call_descriptions;
    };

    /**
     * Answer the module description manager of the factory.
     *
     * @return The module description manager of the factory.
     */
    virtual const ModuleDescriptionManager& GetModuleDescriptionManager() const {
        return module_descriptions;
    };

protected:
    /** Ctor. */
    AbstractObjectFactoryInstance() : call_descriptions(), module_descriptions(){};

    /** Dtor. */
    virtual ~AbstractObjectFactoryInstance() = default;

    /** The call description manager of the factory. */
    CallDescriptionManager call_descriptions;

    /** The module description manager of the factory. */
    ModuleDescriptionManager module_descriptions;
};

} // namespace megamol::core::factories

#endif // MEGAMOLCORE_FACTORIES_ABSTRACTOBJECTFACTORYINSTANCE_H_INCLUDED

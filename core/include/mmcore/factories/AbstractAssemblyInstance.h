/**
 * MegaMol
 * Copyright (c) 2015-2021, MegaMol Dev Team
 * All rights reserved.
 */

#ifndef MEGAMOLCORE_FACTORIES_ABSTRACTASSEMBLYINSTANCE_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_ABSTRACTASSEMBLYINSTANCE_H_INCLUDED
#pragma once

#include <string>

#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescriptionManager.h"

namespace megamol::core::factories {

    /**
     * Abstract base class for all object descriptions.
     *
     * An object is described using a unique name. This name is compared case
     * insensitive!
     */
    class AbstractAssemblyInstance {
    public:
        /** deleted copy ctor */
        AbstractAssemblyInstance(const AbstractAssemblyInstance& src) = delete;

        /** deleted assignment operatior */
        AbstractAssemblyInstance& operator=(const AbstractAssemblyInstance& rhs) = delete;

        /**
         * Answer the (machine-readable) name of the assembly. This usually is
         * The name of the plugin dll/so without prefix and extension.
         *
         * @return The (machine-readable) name of the assembly
         */
        virtual const std::string& GetAssemblyName() const = 0;

        /**
         * Answer the call description manager of the assembly.
         *
         * @return The call description manager of the assembly.
         */
        virtual const CallDescriptionManager& GetCallDescriptionManager() const {
            return call_descriptions;
        };

        /**
         * Answer the module description manager of the assembly.
         *
         * @return The module description manager of the assembly.
         */
        virtual const ModuleDescriptionManager& GetModuleDescriptionManager() const {
            return module_descriptions;
        };

    protected:
        /** Ctor. */
        AbstractAssemblyInstance() : call_descriptions(), module_descriptions(){};

        /** Dtor. */
        virtual ~AbstractAssemblyInstance() = default;

        /** The call description manager of the assembly. */
        CallDescriptionManager call_descriptions;

        /** The module description manager of the assembly. */
        ModuleDescriptionManager module_descriptions;
    };

} // namespace megamol::core::factories

#endif // MEGAMOLCORE_FACTORIES_ABSTRACTASSEMBLYINSTANCE_H_INCLUDED

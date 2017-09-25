/*
 * AbstractPluginInstance.h
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_UTILITY_PLUGINS_ABSTRACTPLUGININSTANCE_H_INCLUDED
#define MEGAMOLCORE_UTILITY_PLUGINS_ABSTRACTPLUGININSTANCE_H_INCLUDED
#pragma once

#include "mmcore/factories/AbstractAssemblyInstance.h"
#include <memory>
#include "vislib/macro_utils.h"


namespace megamol {
namespace core {
namespace utility {
namespace plugins {

    /**
     * Abstract base class for all object descriptions.
     *
     * An object is described using a unique name. This name is compared case
     * insensitive!
     */
    class MEGAMOLCORE_API AbstractPluginInstance : public factories::AbstractAssemblyInstance {
    public:

        /** The shared pointer type to be used */
        typedef std::shared_ptr<AbstractPluginInstance const> ptr_type;

        /**
        * Answer the (machine-readable) name of the assembly. This usually is
        * The name of the plugin dll/so without prefix and extension.
        *
        * @return The (machine-readable) name of the assembly
        */
        virtual const std::string& GetAssemblyName(void) const;

        /**
        * Answer the (human-readable) description of the plugin assembly.
        *
        * @return The (human-readable) description of the plugin assembly
        */
        virtual const std::string& GetDescription(void) const;

    protected:

        /** Ctor. */
        AbstractPluginInstance(const char *asm_name, const char *description);

        /** Dtor. */
        virtual ~AbstractPluginInstance(void);

    private:

        /** deleted copy ctor */
        AbstractPluginInstance(const AbstractPluginInstance& src) = delete;

        /** deleted assignment operatior */
        AbstractPluginInstance& operator=(const AbstractPluginInstance& rhs) = delete;

        /** The (machine-readable) name of the assembly */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        std::string asm_name;

        /** The (human-readable) description of the plugin assembly */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        std::string description;

    };

} /* end namespace plugins */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_UTILITY_PLUGINS_ABSTRACTPLUGININSTANCE_H_INCLUDED */

/*
 * AbstractAssemblyInstance.h
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_FACTORIES_ABSTRACTASSEMBLYINSTANCE_H_INCLUDED
#define MEGAMOLCORE_FACTORIES_ABSTRACTASSEMBLYINSTANCE_H_INCLUDED
#pragma once

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/CallDescriptionManager.h"
#include "mmcore/factories/ModuleDescriptionManager.h"
#include <string>


namespace megamol {
namespace core {
namespace factories {

    /**
     * Abstract base class for all object descriptions.
     *
     * An object is described using a unique name. This name is compared case
     * insensitive!
     */
    class MEGAMOLCORE_API AbstractAssemblyInstance {
    public:

        /**
         * Answer the (machine-readable) name of the assembly. This usually is
         * The name of the plugin dll/so without prefix and extension.
         *
         * @return The (machine-readable) name of the assembly
         */
        virtual const std::string& GetAssemblyName(void) const = 0;

        /**
         * Answer the call description manager of the assembly.
         *
         * @return The call description manager of the assembly.
         */
        virtual const CallDescriptionManager& GetCallDescriptionManager(void) const;

        /**
         * Answer the module description manager of the assembly.
         *
         * @return The module description manager of the assembly.
         */
        virtual const ModuleDescriptionManager& GetModuleDescriptionManager(void) const;

        /**
         * Gets the assembly file name
         *
         * @return The assembly file name
         */
        inline std::string GetAssemblyFileNameA(void) const {
            std::string s;
            GetAssemblyFileName(s);
            return s;
        }

        /**
         * Gets the assembly file name
         *
         * @return The assembly file name
         */
        inline std::wstring GetAssemblyFileNameW(void) const {
            std::wstring s;
            GetAssemblyFileName(s);
            return s;
        }

        /**
         * Gets the assembly file name
         *
         * @param out_filename String variable to receive the file name
         */
        void GetAssemblyFileName(std::string& out_filename) const;

        /**
         * Gets the assembly file name
         *
         * @param out_filename String variable to receive the file name
         */
        void GetAssemblyFileName(std::wstring& out_filename) const;

        /**
         * Sets the assembly file name
         *
         * @param filename The assembly file name
         */
        void SetAssemblyFileName(const std::string& filename);

        /**
         * Sets the assembly file name
         *
         * @param filename The assembly file name
         */
        void SetAssemblyFileName(const std::wstring& filename);

    protected:

        /** Ctor. */
        AbstractAssemblyInstance(void);

        /** Dtor. */
        virtual ~AbstractAssemblyInstance(void);

        /** The call description manager of the assembly. */
        CallDescriptionManager call_descriptions;

        /** The module description manager of the assembly. */
        ModuleDescriptionManager module_descriptions;

    private:

        /** deleted copy ctor */
        AbstractAssemblyInstance(const AbstractAssemblyInstance& src) = delete;

        /** deleted assignment operatior */
        AbstractAssemblyInstance& operator=(const AbstractAssemblyInstance& rhs) = delete;

        /** The assembly file name in utf8 */
        char* filename;

    };

} /* end namespace factories */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_FACTORIES_ABSTRACTASSEMBLYINSTANCE_H_INCLUDED */

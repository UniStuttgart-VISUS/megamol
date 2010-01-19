/*
 * ModuleDescription.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_MODULEDESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_MODULEDESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "api/MegaMolCore.std.h"
#include "ObjectDescription.h"
#include "Module.h"


namespace megamol {
namespace core {

    /** forward declaration */
    class CoreInstance;


    /**
     * Abstract base class of rendering graph module descriptions
     */
    class MEGAMOLCORE_API ModuleDescription : public ObjectDescription {
    public:

        /** Ctor. */
        ModuleDescription(void);

        /** Dtor. */
        virtual ~ModuleDescription(void);

        /**
         * Answer the class name of the module described.
         *
         * @return The class name of the module described.
         */
        virtual const char *ClassName(void) const = 0;

        /**
         * Creates a new module object from this description.
         *
         * @param name The name for the module to be created.
         * @param instance The core instance calling. Must not be 'NULL'.
         *
         * @return The newly created module object or 'NULL' in case of an
         *         error.
         */
        Module *CreateModule(const vislib::StringA& name,
            class ::megamol::core::CoreInstance *instance) const;

        /**
         * Gets a human readable description of the module.
         *
         * @return A human readable description of the module.
         */
        virtual const char *Description(void) const = 0;

        /**
         * Answers whether this module is available on the current system.
         * This implementation always returns 'true'.
         *
         * @return 'true' if the module is available, 'false' otherwise.
         */
        virtual bool IsAvailable(void) const;

        /**
         * Answers whether this description is describing the class of
         * 'module'.
         *
         * @param module The module to test.
         *
         * @return 'true' if 'module' is described by this description,
         *         'false' otherwise.
         */
        virtual bool IsDescribing(const Module * module) const = 0;

    protected:

        /**
         * Creates a new module object from this description.
         *
         * @return The newly created module object or 'NULL' in case of an
         *         error.
         */
        virtual Module *createModuleImpl(void) const = 0;

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_MODULEDESCRIPTION_H_INCLUDED */

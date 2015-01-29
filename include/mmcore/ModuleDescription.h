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

        /**
         * Answers whether this modules is a data source loader supporting
         * file format auto-detection for quickstarting.
         *
         * This default implementation returns 'false'.
         *
         * @return 'true' if the module is a data source loader supporting
         *         file format auto-detection.
         */
        virtual bool IsLoaderWithAutoDetection(void) const;

        /**
         * Answer whether or not this module can be used in a quickstart
         *
         * @return 'true' if the module can be used in a quickstart
         */
        virtual bool IsVisibleForQuickstart(void) const = 0;

        /**
         * Answer the file format file name extensions usually used by files
         * for this loader module, or NULL if there are none. The file name
         * extensions include the periode but no asterix (e. g. '.dat').
         * Multiple extensions are separated by semicolons.
         *
         * This default implementation returns 'NULL'.
         *
         * @return The file name extenions for data files for this loader
         *         module.
         */
        virtual const char *LoaderAutoDetectionFilenameExtensions(void) const;

        /**
         * Performs an file format auto detection check based on the first
         * 'dataSize' bytes of the file.
         *
         * This default implementation returns 0
         *
         * @param data Pointer to the first 'dataSize' bytes of the file to
         *             test
         * @param dataSize The number of valid bytes stored at 'data'
         *
         * @return The confidence if this data file can be loaded with this
         *         module. A value of '1' tells the caller that the data file
         *         can be loaded (as long as the file is not corrupted). A
         *         value of '0' tells that this file cannot be loaded. Any
         *         value inbetween tells that loading this file might fail or
         *         might result in undefined behaviour.
         */
        virtual float LoaderAutoDetection(const unsigned char* data, SIZE_T dataSize) const;

        /**
         * Answer the relative name of the parameter slot for the data file
         * name of the data file to load.
         *
         * This default implementation returns 'NULL'.
         *
         * @return The name of the file name slot
         */
        virtual const char *LoaderFilenameSlotName(void) const;

        /**
         * Answer the file type name (e. g. "Particle Data")
         *
         * This default implementation returns 'NULL'.
         *
         * @return The file type name
         */
        virtual const char *LoaderFileTypeName(void) const;

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

/*
 * LoaderADModuleAutoDescription.h
 *
 * Copyright (C) 2008 by Universitaet Stuttgart (VIS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_LOADERADMODULEAUTODESCRIPTION_H_INCLUDED
#define MEGAMOLCORE_LOADERADMODULEAUTODESCRIPTION_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/ModuleAutoDescription.h"
#include "vislib/sys/Log.h"


namespace megamol {
namespace core {


    /**
     * Class of rendering graph module descriptions generated using static
     * member implementations of the module classes.
     * Template parameter 'C' is the module class to be described.
     *
     * Use this template instead of 'ModuleAutoDescription' if the module 'C'
     * is a data source module supporting file format autodetection.
     *
     * 'C' must implement the static methods:
     *      const char* ClassName();
     *      const char *Description();
     *      bool IsAvailable();
     *      const char *FilenameExtensions();
     *      float FileFormatAutoDetect(const unsigned char* data, SIZE_T dataSize);
     *      const char *FilenameSlotName();
     *      const char *FileTypeName();
     *
     * TODO: This is no longer required when redesigning similiar to "SupportsQuickstart" in MegaMol 0.5
     */
    template<class C> class LoaderADModuleAutoDescription : public ModuleAutoDescription<C> {
    public:

        /** Ctor. */
        LoaderADModuleAutoDescription(void) : ModuleAutoDescription<C>() {
            // intentionally empty
        }

        /** Dtor. */
        virtual ~LoaderADModuleAutoDescription(void) {
            // intentionally empty
        }

        /**
         * Answers whether this modules is a data source loader supporting
         * file format auto-detection for quickstarting.
         *
         * This implementation returns 'true'.
         *
         * @return 'true' if the module is a data source loader supporting
         *         file format auto-detection.
         */
        virtual bool IsLoaderWithAutoDetection(void) const {
            return true;
        }

        /**
         * Answer the file format file name extensions usually used by files
         * for this loader module, or NULL if there are none. The file name
         * extensions include the periode but no asterix (e. g. '.dat').
         * Multiple extensions are separated by semicolons.
         *
         * @return The file name extenions for data files for this loader
         *         module.
         */
        virtual const char *LoaderAutoDetectionFilenameExtensions(void) const {
            return C::FilenameExtensions();
        }

        /**
         * Performs an file format auto detection check based on the first
         * 'dataSize' bytes of the file.
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
        virtual float LoaderAutoDetection(const unsigned char* data, SIZE_T dataSize) const {
            return C::FileFormatAutoDetect(data, dataSize);
        }

        /**
         * Answer the relative name of the parameter slot for the data file
         * name of the data file to load.
         *
         * @return The name of the file name slot
         */
        virtual const char *LoaderFilenameSlotName(void) const {
            return C::FilenameSlotName();
        }

        /**
         * Answer the file type name (e. g. "Particle Data")
         *
         * @return The file type name
         */
        virtual const char *LoaderFileTypeName(void) const {
            const char *fn = C::FileTypeName();
            if (fn == NULL) {
                fn = C::FilenameExtensions();
            }
            return fn;
        }

    protected:

    };


} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_LOADERADMODULEAUTODESCRIPTION_H_INCLUDED */

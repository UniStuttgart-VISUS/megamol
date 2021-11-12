/*
 * ResourceWrapper.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS). 
 * Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_RESOURCEWRAPPER_H_INCLUDED
#define MEGAMOLCORE_RESOURCEWRAPPER_H_INCLUDED
#if (defined(_MSC_VER) && (_MSC_VER > 1000))
#pragma once
#endif /* (defined(_MSC_VER) && (_MSC_VER > 1000)) */

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/Configuration.h"
#include "vislib/Array.h"
#include "vislib/String.h"


namespace megamol {
namespace core {
namespace utility {

    /**
     * Helper class for generic resource handling.
     *
     * Same as shaders, resources can be placed in configurable directories
     * which can then be automatically searched for the required files. 
     */
    class MEGAMOLCORE_API ResourceWrapper {
    public:

        /**
         * Loads a file called name from the first resource directory it
         * can be found in. outData needs to be deallocated by the caller!
         *
         * @param config the configuration of the current MegaMol instance
         *
         * @param name the filename (including extension) that is to be
         *             loaded
         *
         * @param outData pointer to the contents of the file, if
         *                successfully loaded, otherwise NULL
         *
         * @return size in bytes of the file contents, 0 if loading failed
         */
        static SIZE_T LoadResource(const Configuration& config, 
            const vislib::StringA & name, void **outData);

        /**
         * Loads a file called name from the first resource directory it
         * can be found in and terminates the string. outData needs to be
         * deallocated by the caller!
         *
         * @param config the configuration of the current MegaMol instance
         *
         * @param name the filename (including extension) that is to be
         *             loaded
         *
         * @param outData pointer to the contents of the file, if
         *                successfully loaded, otherwise NULL
         *
         * @return size in bytes of the file contents, 0 if loading failed
         */
        static SIZE_T LoadTextResource(const Configuration& config, 
            const vislib::StringA & name, char **outData);

        /**
         * Returns the path of the first file in the resource
         * directories that matches the given name.
         *
         * @param config the configuration of the current MegaMol instance
         *
         * @param name the filename (including extension) that is to be
         *             loaded
         *
         * @return the concrete file name
         */
        static vislib::StringW getFileName(const Configuration& config, 
            const vislib::StringA & name);
    };

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_SHADERSOURCEFACTORY_H_INCLUDED */

/*
 * ResourceWrapper.h
 *
 * Copyright (C) 2012 by Universitaet Stuttgart (VISUS).
 * Alle Rechte vorbehalten.
 */

#pragma once

#include "RuntimeConfig.h"
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
class ResourceWrapper {
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
    static SIZE_T LoadResource(
        frontend_resources::RuntimeConfig const& runtimeConf, const vislib::StringA& name, void** outData);

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
    static SIZE_T LoadTextResource(
        frontend_resources::RuntimeConfig const& runtimeConf, const vislib::StringA& name, char** outData);

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
    static vislib::StringW getFileName(
        frontend_resources::RuntimeConfig const& runtimeConf, const vislib::StringA& name);
};

} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

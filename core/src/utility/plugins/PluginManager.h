/*
 * PluginManager.h
 * Copyright (C) 2015 by MegaMol Consortium
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_UTILITY_PLUGIN_PLUGINMANAGER_H_INCLUDED
#define MEGAMOLCORE_UTILITY_PLUGIN_PLUGINMANAGER_H_INCLUDED
#pragma once

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include <vector>
#include "vislib/tchar.h"
#include "vislib/sys/DynamicLinkLibrary.h"
#include <string>


namespace megamol {
namespace core {

    /* forward declaration */
    class CoreInstance;

namespace utility {
namespace plugins {

    /**
     * Abstract base class for all object descriptions.
     *
     * An object is described using a unique name. This name is compared case
     * insensitive!
     */
    class PluginManager {
    public:

        /** Type of pointers to loaded plugins */
        typedef AbstractPluginInstance::ptr_type plugin_ptr_type;

        /** Type of list managing loaded plugins */
        typedef std::vector<plugin_ptr_type> collection_type;

        /** Ctor. */
        PluginManager(void);

        /** Dtor. */
        ~PluginManager(void);

        /**
         * Loads a plugin file. This might also load dependent plugins
         *
         * @param filename The path to the plugin file to load
         * @param coreInst The CoreInstance calling. This must always be the
         *                 same object!
         *
         * @return A collection of pointers to all newly loaded plugins. This
         *         collection at least holds the requested plugin, but might
         *         also hold dependent plugins which got loaded as well.
         *
         * @throw std::exception in case of an error.
         */
        collection_type LoadPlugin(const std::basic_string<TCHAR>& filename,
            ::megamol::core::CoreInstance& coreInst);

        /**
         * Answer the collection of loaded plugins
         *
         * @return The loaded plugins
         */
        inline const collection_type& GetPlugins(void) const {
            return this->plugins;
        }

    private:

        /** deleted copy ctor */
        PluginManager(const PluginManager& src) = delete;

        /** deleted assignment operatior */
        PluginManager& operator=(const PluginManager& rhs) = delete;

        /**
         * Continue to load 'lib' as plugin of version 2.00
         *
         * @param path The plugin's file system path
         * @param lib The loaded library
         * @param coreInst The calling core instance
         *
         * @return A collection containing the loaded plugin
         *
         * @throw vislib::Exception on failure
         */
        collection_type ContinueLoad200(
            const std::basic_string<TCHAR> &path,
            std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib,
            CoreInstance& coreInst);


        /** The loaded plugins */
        collection_type plugins;

    };

} /* end namespace plugins */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_UTILITY_PLUGIN_PLUGINMANAGER_H_INCLUDED */

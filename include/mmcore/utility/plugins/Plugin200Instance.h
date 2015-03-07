/*
 * Plugin200Instance.h
 * Copyright (C) 2015 by MegaMol Team
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED
#define MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED
#pragma once

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "vislib/sys/DynamicLinkLibrary.h"
#include "vislib/macro_utils.h"
#include <memory>
#include "mmcore/utility/plugins/Plugin200Utilities.h"


namespace megamol {
namespace core {
namespace utility {
namespace plugins {

    /**
     * Struct holding version information about a single dependent library
     */
    typedef struct _library_version_info_t {

        /** The (maschine-readable) name of the library */
        const char *name;

        /** The number of version number elements */
        unsigned short version_len;

        /** The version number elements, starting with the most significant element */
        unsigned short *version;

    } LibraryVersionInfo;

    /**
     * Struct holding all compatibility information to dependent libraries
     */
    typedef struct _plugin_compat_info_t {

        /** The number of library version info structs */
        unsigned int libs_cnt;

        /** The library version info structs */
        LibraryVersionInfo *libs;

    } PluginCompatibilityInfo;

    /** Callback to be fired on an error during plugin initialization */
    typedef void (*ErrorCallback)(const char *msg, const char *file, unsigned int line);

    /**
     * Base class for Instances of Plugins using the 2.0 API interface
     */
    class MEGAMOLCORE_API Plugin200Instance : public AbstractPluginInstance {
    public:

        /** Type declaration of 'mmplgPluginAPIVersion' */
        typedef int (*mmplgPluginAPIVersion_funcptrtype)(void);

        /** Type declaration of 'mmplgGetPluginCompatibilityInfo' */
        typedef PluginCompatibilityInfo *(*mmplgGetPluginCompatibilityInfo_funcptrtype)(ErrorCallback onError);

        /** Type declaration of 'mmplgReleasePluginCompatibilityInfo' */
        typedef void(*mmplgReleasePluginCompatibilityInfo_funcptrtype)(PluginCompatibilityInfo*);

        /** Type declaration of 'mmplgGetPluginInstance' */
        typedef AbstractPluginInstance* (*mmplgGetPluginInstance_funcptrtype)(ErrorCallback onError);

        /** Type declaration of 'mmplgReleasePluginInstance' */
        typedef void(*mmplgReleasePluginInstance_funcptrtype)(AbstractPluginInstance*);

        /** Possible target values for static connectors */
        enum class StaticConnectorType {
            Log,
            StackTrace
        };

        /** Dtor */
        virtual ~Plugin200Instance(void);

        /**
         * TODO: Document
         */
        virtual void registerClasses(void) = 0;

        /**
         * TODO: Document
         */
        virtual void connectStatics(StaticConnectorType which, void* value) = 0;

        /**
         * Stores the lib object as part of this instance
         *
         * @param lib The lib object to be stored
         *
         * @remarks This method is for framework management only. DO NOT CALL!
         */
        void store_lib(std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib);

        /**
         * Answer the stored lib
         *
         * @return The stored lib
         */
        std::shared_ptr<vislib::sys::DynamicLinkLibrary> get_lib(void) const;

    protected:

        /**
         * Ctor
         *
         * @param asm_name The (machine-readable) name of the assembly
         * @param description 
         */
        Plugin200Instance(const char *asm_name, const char *description);

    private:

        /* deleted copy ctor */
        Plugin200Instance(const Plugin200Instance& src) = delete;

        /* deleted assignment operator */
        Plugin200Instance& operator=(const Plugin200Instance& src) = delete;

        /** The plugin library object */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib;

    };


} /* end namespace plugins */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#endif /* MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED */

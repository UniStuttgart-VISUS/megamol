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

        /** Additional flags specifying options */
        unsigned int flags;

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

        /** Type declaration to register something at the core instance*/
        typedef void(*registerAtCoreInstance_funcptrtype)(CoreInstance& inst);

        /** Dtor */
        virtual ~Plugin200Instance(void);

        /**
         * Answer the call description manager of the assembly.
         *
         * @return The call description manager of the assembly.
         */
        virtual const factories::CallDescriptionManager& GetCallDescriptionManager(void) const;

        /**
         * Answer the module description manager of the assembly.
         *
         * @return The module description manager of the assembly.
         */
        virtual const factories::ModuleDescriptionManager& GetModuleDescriptionManager(void) const;

        /**
         * Connect static objects of the core with their counterparts in the
         * plugin instance.
         *
         * @param which Id of the static object to be connected
         * @param value The value to connect to
         *
         * @remark Do not implement this method manually, but use the macro
         * 'MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics'
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

        /**
         * Calls all functions which want to register something at the core instance.
         * Also clears that list.
         *
         * @param core The core instance
         */
        void callRegisterAtCoreInstanceFunctions(CoreInstance& core);

    protected:

        /**
         * This factory methode registers all module and call classes exported
         * by this plugin instance at the respective factories.
         *
         * @remarks This method is automatically called when the factories are
         *          accessed for the first time. Do not call manually.
         */
        virtual void registerClasses(void) = 0;

        /**
         * Ctor
         *
         * @param asm_name The (machine-readable) name of the assembly
         * @param description 
         */
        Plugin200Instance(const char *asm_name, const char *description);

        /**
         * Adds a function to register something at the core instance
         *
         * @param func The function to register.
         */
        void addRegisterAtCoreInstanceFunction(registerAtCoreInstance_funcptrtype func);

    private:

        /** Ensures that registered classes was called */
        void ensureRegisterClassesWrapper(void) const;

        /* deleted copy ctor */
        Plugin200Instance(const Plugin200Instance& src) = delete;

        /* deleted assignment operator */
        Plugin200Instance& operator=(const Plugin200Instance& src) = delete;

        /** The plugin library object */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        std::shared_ptr<vislib::sys::DynamicLinkLibrary> lib;

        /** Functions to be called to register something at the core instance */
        VISLIB_MSVC_SUPPRESS_WARNING(4251)
        std::vector<registerAtCoreInstance_funcptrtype> regAtCoreFuncs;

        /** 
         * Flag whether or not the module and call classes have been registered
         */
        bool classes_registered;

    };


} /* end namespace plugins */
} /* end namespace utility */
} /* end namespace core */
} /* end namespace megamol */

#include "mmcore/utility/plugins/Plugin200Utilities.h"

#endif /* MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED */

/*
 * Plugin200Utilites.h
 * Copyright (C) 2015 by MegaMol Team
 * All rights reserved. Alle Rechte vorbehalten.
 */

#ifndef MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200UTILITIES_H_INCLUDED
#define MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200UTILITIES_H_INCLUDED
#pragma once

#ifndef MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED
#error Do not include "Plugin200Utilites.h" directly. Instead include "Plugin200Instance.h"
#endif


/*
 * These macros should be used to simplify the implementation of the plugin
 * v2.0 api of MegaMol
 */


/** Flag marking that this is a normal build (release, clean) */
#define MEGAMOLCORE_PLUGIN200UTIL_FLAGS_NONE        0x00000000

/** Flag marking that this is a debug build */
#define MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD 0x00000001

/** Flag marking that this is a dirty build (unclean repository working copy state) */
#define MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD 0x00000002

/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
 */
#define MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion return 200;


#include <string>

namespace megamol {
namespace core{
namespace utility {
namespace plugins {

    /**
     * Sets the LibraryVersionInfo fields
     *
     * @param info The LibraryVersionInfo object
     * @param name The library name
     * @param v1 The version number component 1
     * @param v2 The version number component 2
     * @param v3 The version number component 3
     * @param v4 The version number component 4
     * @param flags The build flags
     */
    template<class T1, class T2, class T3>
    inline void SetLibraryVersionInfo(LibraryVersionInfo &info,
            const char* name, 
            T1 v1, T2 v2, T3 v3,
            unsigned int flags = MEGAMOLCORE_PLUGIN200UTIL_FLAGS_NONE) {
        std::string n(name);
        char *nm = new char[n.length() + 1];
        info.name = nm;
        ::memcpy(nm, n.c_str(), n.length() + 1);
        info.version.resize(3);
        info.version[0] = reinterpret_cast<const char*>(v1);
        info.version[1] = reinterpret_cast<const char*>(v2);
        info.version[2] = v3;
        info.version_len = 3;
        info.flags = flags;
    }

}
}
}
}


/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
 */
#define MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci) \
    if (ci == nullptr) return; \
    for (unsigned int i = 0; i < ci->libs_cnt; i++) { \
        delete[] ci->libs[i].name; \
        ci->libs[i].version.clear(); \
        ci->libs[i].name = nullptr; \
        ci->libs[i].version_len = 0; \
    } \
    delete[] ci->libs; \
    ci->libs = nullptr; \
    ci->libs_cnt = 0; \
    delete ci;

#include "vislib/sys/Log.h"

/*
 * Usage
 * class plugin_instance : public Plugin200Instance {
 *     ...
 *     MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
 *     ...
 * };
 */
#define MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics \
    virtual void connectStatics(StaticConnectorType which, void* value) { \
        switch (which) { \
            case StaticConnectorType::Log: \
                vislib::sys::Log::DefaultLog.SetLogFileName(static_cast<const char*>(nullptr), false); \
                vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_NONE); \
                vislib::sys::Log::DefaultLog.SetEchoTarget(new vislib::sys::Log::RedirectTarget(static_cast<vislib::sys::Log*>(value))); \
                vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL); \
                vislib::sys::Log::DefaultLog.EchoOfflineMessages(true); \
                break; \
        } \
    }

#include "vislib/Exception.h"
#include <exception>

/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
 */
#define MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError) \
    try { \
        return new plugin_instance(); \
    } catch(vislib::Exception& ex) { \
        if (onError) onError(ex.GetMsgA(), ex.GetFile(), ex.GetLine()); \
    } catch(std::exception& ex) { \
        if (onError) onError(ex.what(), __FILE__, __LINE__); \
    } catch(...) { \
        if (onError) onError("Unknown exception", __FILE__, __LINE__); \
    } \
    return nullptr;

/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
 */
#define MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi) \
    plugin_instance *i = dynamic_cast<plugin_instance*>(pi); \
    if (i != nullptr) { \
        delete i; \
    }
    // ensure 'pi' is an instance of that plugin
    // deleting 'i' it here ensures the use of the correct heap!
    // "delete pi" would be wrong, as it would call the right dtor, but on
    // the wrong heap controller.

#endif /* MEGAMOLCORE_UTILITY_PLUGINS_PLUGIN200INSTANCE_H_INCLUDED */

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


/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
 */
#define MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion return 200;

/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_Set_LibraryVersionInfo_V4(ci->libs[0], "MegaMolCore", MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_MAJOR_REV, MEGAMOL_CORE_MINOR_REV)
 */
#define MEGAMOLCORE_PLUGIN200UTIL_Set_LibraryVersionInfo_V4(Obj, Name, V1, V2, V3, V4) { \
    const char libname[] = Name; \
    const int libnamelen = sizeof(libname) + 1; \
    char *str; \
    Obj.name = str = new char[libnamelen]; \
    ::memcpy(str, libname, libnamelen); \
    Obj.version_len = 4; \
    Obj.version = new unsigned short[4]; \
    Obj.version[0] = V1; \
    Obj.version[1] = V2; \
    Obj.version[2] = V3; \
    Obj.version[3] = V4; \
}

/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_Set_LibraryVersionInfo_V5(ci->libs[0], "MegaMolCore", MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_MAJOR_REV, MEGAMOL_CORE_MINOR_REV, MEGAMOL_CORE_ISDIRTY)
 */
#define MEGAMOLCORE_PLUGIN200UTIL_Set_LibraryVersionInfo_V5(Obj, Name, V1, V2, V3, V4, V5) { \
    const char libname[] = Name; \
    const int libnamelen = sizeof(libname) + 1; \
    char *str; \
    Obj.name = str = new char[libnamelen]; \
    ::memcpy(str, libname, libnamelen); \
    Obj.version_len = 5; \
    Obj.version = new unsigned short[5]; \
    Obj.version[0] = V1; \
    Obj.version[1] = V2; \
    Obj.version[2] = V3; \
    Obj.version[3] = V4; \
    Obj.version[4] = V5; \
}

/*
 * Usage:
 *   MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
 */
#define MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci) \
    if (ci == nullptr) return; \
    for (unsigned int i = 0; i < ci->libs_cnt; i++) { \
        delete[] ci->libs[i].name; \
        delete[] ci->libs[i].version; \
        ci->libs[i].name = nullptr; \
        ci->libs[i].version = nullptr; \
        ci->libs[i].version_len = 0; \
    } \
    delete[] ci->libs; \
    ci->libs = nullptr; \
    ci->libs_cnt = 0; \
    delete ci;

#include "vislib/sys/Log.h"
#include "vislib/sys/ThreadSafeStackTrace.h"

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
            case StaticConnectorType::StackTrace: \
                vislib::sys::ThreadSafeStackTrace::Initialise( \
                    *static_cast<const vislib::SmartPtr<vislib::StackTrace>*>(value), true); \
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

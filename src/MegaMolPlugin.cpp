/*
 * MegaMolPlugin.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "MegaMolPlugin/MegaMolPlugin.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/ModuleDescription.h"
#include "vislib/vislibversion.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/ThreadSafeStackTrace.h"


/*
 * mmplgPluginAPIVersion
 */
MEGAMOLPLUGIN_API int mmplgPluginAPIVersion(void) {
    return 100;
}


/*
 * mmplgPluginName
 */
MEGAMOLPLUGIN_API const char * mmplgPluginName(void) {
    return "PluginTemplate";
}


/*
 * mmplgPluginDescription
 */
MEGAMOLPLUGIN_API const char * mmplgPluginDescription(void) {
    return "Template for MegaMol Plugins (TODO: CHANGE this description)";
}


/*
 * mmplgCoreCompatibilityValue
 */
MEGAMOLPLUGIN_API const void * mmplgCoreCompatibilityValue(void) {
    static const mmplgCompatibilityValues compRev = {
        sizeof(mmplgCompatibilityValues),
        MEGAMOL_CORE_COMP_REV,
        VISLIB_VERSION_REVISION
    };
    return &compRev;
}


/*
 * mmplgModuleCount
 */
MEGAMOLPLUGIN_API int mmplgModuleCount(void) {
    return 0; // TODO: Implement
}


/*
 * mmplgModuleDescription
 */
MEGAMOLPLUGIN_API void* mmplgModuleDescription(int idx) {
    return NULL; // TODO: Implement
}


/*
 * mmplgCallCount
 */
MEGAMOLPLUGIN_API int mmplgCallCount(void) {
    return 0; // TODO: Implement
}


/*
 * mmplgCallDescription
 */
MEGAMOLPLUGIN_API void* mmplgCallDescription(int idx) {
    return NULL; // TODO: Implement
}


/*
 * mmplgConnectStatics
 */
MEGAMOLPLUGIN_API bool mmplgConnectStatics(int which, void* value) {
    switch (which) {

        case 1: // vislib::log
            vislib::sys::Log::DefaultLog.SetLogFileName(static_cast<const char*>(NULL), false);
            vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_NONE);
            vislib::sys::Log::DefaultLog.SetEchoTarget(new vislib::sys::Log::RedirectTarget(static_cast<vislib::sys::Log*>(value)));
            vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
            vislib::sys::Log::DefaultLog.EchoOfflineMessages(true);
            return true;

        case 2: // vislib::stacktrace
            return vislib::sys::ThreadSafeStackTrace::Initialise(
                *static_cast<const vislib::SmartPtr<vislib::StackTrace>*>(value), true);

    }
    return false;
}

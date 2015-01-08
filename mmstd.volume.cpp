/*
 * mmstd.volume.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmstd.volume.h"
#include "api/MegaMolCore.std.h"
#include "ModuleAutoDescription.h"
#include "vislib/vislibversion.h"
#include "vislib/Log.h"
#include "vislib/ThreadSafeStackTrace.h"
#include "DatRawDataSource.h"
#include "DirectVolumeRenderer.h"


/*
 * mmplgPluginAPIVersion
 */
MMSTD_VOLUME_API int mmplgPluginAPIVersion(void) {
    return 100;
}


/*
 * mmplgPluginName
 */
MMSTD_VOLUME_API const char * mmplgPluginName(void) {
    return "mmstd.volume";
}


/*
 * mmplgPluginDescription
 */
MMSTD_VOLUME_API const char * mmplgPluginDescription(void) {
    return "MegaMol Plugins for volume data modules";
}


/*
 * mmplgCoreCompatibilityValue
 */
MMSTD_VOLUME_API const void * mmplgCoreCompatibilityValue(void) {
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
MMSTD_VOLUME_API int mmplgModuleCount(void) {
    return 2;
}


/*
 * mmplgModuleDescription
 */
MMSTD_VOLUME_API void* mmplgModuleDescription(int idx) {
    switch (idx) {
    case 0: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::volume::DatRawDataSource>();
    case 1: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::volume::DirectVolumeRenderer>();
    }
    return NULL;
}


/*
 * mmplgCallCount
 */
MMSTD_VOLUME_API int mmplgCallCount(void) {
    return 0;
}


/*
 * mmplgCallDescription
 */
MMSTD_VOLUME_API void* mmplgCallDescription(int idx) {
    return NULL;
}


/*
 * mmplgConnectStatics
 */
MMSTD_VOLUME_API bool mmplgConnectStatics(int which, void* value) {
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

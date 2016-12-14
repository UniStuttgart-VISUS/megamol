/*
 * mmstd.volume.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmstd_volume/mmstd_volume.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "vislib/vislibversion.h"
#include "vislib/sys/Log.h"
#include "DatRawDataSource.h"
#include "DirectVolumeRenderer.h"
#include "BuckyBall.h"
#include "GridBalls.h"
#include "DirPartVolume.h"
#include "VolumeCache.h"
#include "RenderVolumeSlice.h"


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
    return 7;
}


/*
 * mmplgModuleDescription
 */
MMSTD_VOLUME_API void* mmplgModuleDescription(int idx) {
    switch (idx) {
    case 0: return new megamol::core::factories::ModuleAutoDescription<megamol::stdplugin::volume::DatRawDataSource>();
    case 1: return new megamol::core::factories::ModuleAutoDescription<megamol::stdplugin::volume::DirectVolumeRenderer>();
    case 2: return new megamol::core::factories::ModuleAutoDescription<megamol::stdplugin::volume::BuckyBall>();
    case 3: return new megamol::core::factories::ModuleAutoDescription<megamol::stdplugin::volume::GridBalls>();
    case 4: return new megamol::core::factories::ModuleAutoDescription<megamol::stdplugin::volume::DirPartVolume>();
    case 5: return new megamol::core::factories::ModuleAutoDescription<megamol::stdplugin::volume::VolumeCache>();
    case 6: return new megamol::core::factories::ModuleAutoDescription<megamol::stdplugin::volume::RenderVolumeSlice>();
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
            return true;

    }
    return false;
}

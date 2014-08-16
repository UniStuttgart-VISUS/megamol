/*
 * mmstd.datatools.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmstd.datatools.h"
#include "api/MegaMolCore.std.h"
#include "ModuleAutoDescription.h"
#include "vislib/vislibversion.h"
#include "vislib/Log.h"
#include "vislib/ThreadSafeStackTrace.h"
#include "DataSetTimeRewriteModule.h"
#include "ParticleListMergeModule.h"
#include "DataFileSequencer.h"
#include "SphereDataUnifier.h"
#include "ParticleThinner.h"
#include "OverrideParticleGlobals.h"
#include "ParticleRelaxationModule.h"


/*
 * mmplgPluginAPIVersion
 */
MMSTD_DATATOOLS_API int mmplgPluginAPIVersion(void) {
    return 100;
}


/*
 * mmplgPluginName
 */
MMSTD_DATATOOLS_API const char * mmplgPluginName(void) {
    return "mmstd.datatools";
}


/*
 * mmplgPluginDescription
 */
MMSTD_DATATOOLS_API const char * mmplgPluginDescription(void) {
    return "MegaMol Standard-Plugin containing data manipulation and conversion modules";
}


/*
 * mmplgCoreCompatibilityValue
 */
MMSTD_DATATOOLS_API const void * mmplgCoreCompatibilityValue(void) {
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
MMSTD_DATATOOLS_API int mmplgModuleCount(void) {
    return 7;
}


/*
 * mmplgModuleDescription
 */
MMSTD_DATATOOLS_API void* mmplgModuleDescription(int idx) {
    switch(idx) {
    case 0: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::datatools::DataSetTimeRewriteModule>();
    case 1: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::datatools::ParticleListMergeModule>();
    case 2: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::datatools::DataFileSequencer>();
    case 3: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::datatools::SphereDataUnifier>();
    case 4: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::datatools::ParticleThinner>();
    case 5: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::datatools::OverrideParticleGlobals>();
    case 6: return new megamol::core::ModuleAutoDescription<megamol::stdplugin::datatools::ParticleRelaxationModule>();
    }
    return nullptr;
}


/*
 * mmplgCallCount
 */
MMSTD_DATATOOLS_API int mmplgCallCount(void) {
    return 0;
}


/*
 * mmplgCallDescription
 */
MMSTD_DATATOOLS_API void* mmplgCallDescription(int idx) {
    return nullptr;
}


/*
 * mmplgConnectStatics
 */
MMSTD_DATATOOLS_API bool mmplgConnectStatics(int which, void* value) {
    switch (which) {

        case 1: // vislib::log
            vislib::sys::Log::DefaultLog.SetLogFileName(static_cast<const char*>(nullptr), false);
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

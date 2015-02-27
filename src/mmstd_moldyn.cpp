/*
 * mmstd.moldyn.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmstd_moldyn/mmstd_moldyn.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/factories/LoaderADModuleAutoDescription.h"
#include "vislib/vislibversion.h"
#include "vislib/sys/Log.h"
#include "vislib/sys/ThreadSafeStackTrace.h"
#include "io/IMDAtomDataSource.h"
#include "io/MMSPDDataSource.h"
#include "io/SIFFDataSource.h"
#include "io/SIFFWriter.h"
#include "io/VIMDataSource.h"
#include "io/VisIttDataSource.h"
#include "rendering/NGSphereRenderer.h"


/*
 * mmplgPluginAPIVersion
 */
MMSTD_MOLDYN_API int mmplgPluginAPIVersion(void) {
    return 100;
}


/*
 * mmplgPluginName
 */
MMSTD_MOLDYN_API const char * mmplgPluginName(void) {
    return "mmstd.moldyn";
}


/*
 * mmplgPluginDescription
 */
MMSTD_MOLDYN_API const char * mmplgPluginDescription(void) {
    return "MegaMol Plugins for Molecular Dynamics Data Visualization";
}


/*
 * mmplgCoreCompatibilityValue
 */
MMSTD_MOLDYN_API const void * mmplgCoreCompatibilityValue(void) {
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
MMSTD_MOLDYN_API int mmplgModuleCount(void) {
    return 7;
}


/*
 * mmplgModuleDescription
 */
MMSTD_MOLDYN_API void* mmplgModuleDescription(int idx) {
    switch (idx) {
    case 0: return new ::megamol::core::factories::LoaderADModuleAutoDescription<::megamol::stdplugin::moldyn::io::IMDAtomDataSource>();
    case 1: return new ::megamol::core::factories::LoaderADModuleAutoDescription<::megamol::stdplugin::moldyn::io::MMSPDDataSource>();
    case 2: return new ::megamol::core::factories::ModuleAutoDescription<::megamol::stdplugin::moldyn::io::SIFFDataSource>();
    case 3: return new ::megamol::core::factories::ModuleAutoDescription<::megamol::stdplugin::moldyn::io::SIFFWriter>();
    case 4: return new ::megamol::core::factories::ModuleAutoDescription<::megamol::stdplugin::moldyn::io::VIMDataSource>();
    case 5: return new ::megamol::core::factories::ModuleAutoDescription<::megamol::stdplugin::moldyn::io::VisIttDataSource>();
	case 6: return new ::megamol::core::factories::ModuleAutoDescription<::megamol::stdplugin::moldyn::rendering::NGSphereRenderer>();
    }
    return NULL;
}


/*
 * mmplgCallCount
 */
MMSTD_MOLDYN_API int mmplgCallCount(void) {
    return 0;
}


/*
 * mmplgCallDescription
 */
MMSTD_MOLDYN_API void* mmplgCallDescription(int idx) {
    return NULL;
}


/*
 * mmplgConnectStatics
 */
MMSTD_MOLDYN_API bool mmplgConnectStatics(int which, void* value) {
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

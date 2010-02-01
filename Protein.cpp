/*
 * Protein.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "Protein.h"
#include "api/MegaMolCore.std.h"

#include "ProteinRendererCartoon.h"
#include "ProteinRenderer.h"
#include "ProteinRendererSES.h"

#include "ProteinData.h"
#include "NetCDFData.h"

#include "CallProteinData.h"
#include "CallFrame.h"

#include "CallAutoDescription.h"
#include "ModuleAutoDescription.h"
#include "vislib/vislibversion.h"

#include "vislib/Log.h"
#include "vislib/ThreadSafeStackTrace.h"

#include "SolPathDataCall.h"
#include "SolPathDataSource.h"
#include "SolPathRenderer.h"


/*
 * mmplgPluginAPIVersion
 */
PROTEIN_API int mmplgPluginAPIVersion(void) {
    return 100;
}


/*
 * mmplgPluginName
 */
PROTEIN_API const char * mmplgPluginName(void) {
    return "Protein";
}


/*
 * mmplgPluginDescription
 */
PROTEIN_API const char * mmplgPluginDescription(void) {
    return "Plugin for protein rendering (SFB716 D4)";
}


/*
 * mmplgCoreCompatibilityValue
 */
PROTEIN_API const void * mmplgCoreCompatibilityValue(void) {
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
PROTEIN_API int mmplgModuleCount(void) {
#if (defined(WITH_NETCDF) && (WITH_NETCDF))
    return 7;
#else
    return 6;
#endif /* (defined(WITH_NETCDF) && (WITH_NETCDF)) */
}


/*
 * mmplgModuleDescription
 */
PROTEIN_API void* mmplgModuleDescription(int idx) {
    switch (idx) {
        case 0: return new megamol::core::ModuleAutoDescription<megamol::core::protein::ProteinData>();
        case 1: return new megamol::core::ModuleAutoDescription<megamol::core::protein::ProteinRenderer>();
        case 2: return new megamol::core::ModuleAutoDescription<megamol::core::protein::ProteinRendererCartoon>();
        case 3: return new megamol::core::ModuleAutoDescription<megamol::core::protein::ProteinRendererSES>();
        case 4: return new megamol::core::ModuleAutoDescription<megamol::protein::SolPathDataSource>();
        case 5: return new megamol::core::ModuleAutoDescription<megamol::protein::SolPathRenderer>();
#if (defined(WITH_NETCDF) && (WITH_NETCDF))
        case 6: return new megamol::core::ModuleAutoDescription<megamol::core::protein::NetCDFData>();
#endif /* (defined(WITH_NETCDF) && (WITH_NETCDF)) */
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgCallCount
 */
PROTEIN_API int mmplgCallCount(void) {
    return 3;
}


/*
 * mmplgCallDescription
 */
PROTEIN_API void* mmplgCallDescription(int idx) {
    switch (idx) {
        case 0: return new megamol::core::CallAutoDescription<megamol::core::protein::CallProteinData>();
        case 1: return new megamol::core::CallAutoDescription<megamol::core::protein::CallFrame>();
        case 2: return new megamol::core::CallAutoDescription<megamol::protein::SolPathDataCall>();
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgConnectStatics
 */
PROTEIN_API bool mmplgConnectStatics(int which, void* value) {
    static vislib::sys::Log::EchoTargetRedirect etr(NULL);
    switch (which) {

        case 1: // vislib::log
            etr.SetTarget(static_cast<vislib::sys::Log*>(value));
            vislib::sys::Log::DefaultLog.SetLogFileName(static_cast<const char*>(NULL), false);
            vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_NONE);
            vislib::sys::Log::DefaultLog.SetEchoOutTarget(&etr);
            vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
            vislib::sys::Log::DefaultLog.EchoOfflineMessages(true);
            return true;

        case 2: // vislib::stacktrace
            return vislib::sys::ThreadSafeStackTrace::Initialise(
                *static_cast<const vislib::SmartPtr<vislib::StackTrace>*>(value), true);

    }
    return false;
}

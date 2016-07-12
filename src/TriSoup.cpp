/*
 * TriSoup.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "TriSoup/TriSoup.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "TriSoupRenderer.h"
#include "TriSoupDataSource.h"
#include "WavefrontObjDataSource.h"
#include "BlockVolumeMesh.h"
#include "volumetrics/VoluMetricJob.h"
#include "OSCBFix.h"
#include "CoordSysMarker.h"
#include "TrackerRendererTransform.h"
#include "volumetrics/IsoSurface.h"
#include "CallBinaryVolumeData.h"
#include "CallVolumetricData.h"
#include "ScreenSpaceEdgeRenderer.h"
#include "vislib/Trace.h"


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "TriSoup",

                /* human-readable plugin description */
                "Plugin for rendering TriSoup mesh data") {

            // here we could perform addition initialization
            vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL - 1);
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::TriSoupRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::TriSoupDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::WavefrontObjDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::BlockVolumeMesh>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::volumetrics::VoluMetricJob>();
            this->module_descriptions.RegisterAutoDescription<megamol::quartz::OSCBFix>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::CoordSysMarker>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::TrackerRendererTransform>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::volumetrics::IsoSurface>();
            this->module_descriptions.RegisterAutoDescription<megamol::trisoup::ScreenSpaceEdgeRenderer>();

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::trisoup::CallBinaryVolumeData>();
            this->call_descriptions.RegisterAutoDescription<megamol::trisoup::CallVolumetricData>();

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
TRISOUP_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
TRISOUP_API
::megamol::core::utility::plugins::PluginCompatibilityInfo *
mmplgGetPluginCompatibilityInfo(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    // compatibility information with core and vislib
    using ::megamol::core::utility::plugins::PluginCompatibilityInfo;
    using ::megamol::core::utility::plugins::LibraryVersionInfo;

    PluginCompatibilityInfo *ci = new PluginCompatibilityInfo;
    ci->libs_cnt = 2;
    ci->libs = new LibraryVersionInfo[2];

    SetLibraryVersionInfo(ci->libs[0], "MegaMolCore",
        MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_MAJOR_REV, MEGAMOL_CORE_MINOR_REV, 0
#if defined(DEBUG) || defined(_DEBUG)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(MEGAMOL_CORE_DIRTY) && (MEGAMOL_CORE_DIRTY != 0)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
        );

    SetLibraryVersionInfo(ci->libs[1], "vislib",
        VISLIB_VERSION_MAJOR, VISLIB_VERSION_MINOR, VISLIB_VERSION_REVISION, VISLIB_VERSION_BUILD, 0
#if defined(DEBUG) || defined(_DEBUG)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(VISLIB_DIRTY_BUILD) && (VISLIB_DIRTY_BUILD != 0)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
        );
    //
    // If you want to test additional compatibilties, add the corresponding versions here
    //

    return ci;
}


/*
 * mmplgReleasePluginCompatibilityInfo
 */
TRISOUP_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
TRISOUP_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
TRISOUP_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

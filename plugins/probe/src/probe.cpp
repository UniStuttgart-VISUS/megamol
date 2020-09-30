/*
 * probe.cpp
 * Copyright (C) 2019 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "probe/probe.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"
#include "ExtractMesh.h"
#include "PlaceProbes.h"
#include "SampleAlongProbes.h"
#include "ProbeCalls.h"
#include "ExtractProbeGeometry.h"
#include "CallKDTree.h"
#include "GenerateGlyphs.h"
#include "SurfaceNets.h"
#include "ExtractCenterline.h"
#include "ConstructKDTree.h"
//#include "ManipulateMesh.h"
#ifdef PROBE_HAS_OSPRAY
#include "OSPRayGlyphGeometry.h"
#endif


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "probe",

                /* human-readable plugin description */
                "Putting probes into data and render glyphs at the end of the probe.") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractMesh>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::PlaceProbes>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::SampleAlongPobes>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractProbeGeometry>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::GenerateGlyphs>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::SurfaceNets>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ExtractCenterline>();
            this->module_descriptions.RegisterAutoDescription<megamol::probe::ConstructKDTree>();
            //this->module_descriptions.RegisterAutoDescription<megamol::probe::ManipulateMesh>();
#ifdef PROBE_HAS_OSPRAY
            this->module_descriptions.RegisterAutoDescription<megamol::probe::OSPRayGlyphGeometry>();
#endif

            // register calls here:

            this->call_descriptions.RegisterAutoDescription<megamol::probe::CallProbes>();
            this->call_descriptions.RegisterAutoDescription<megamol::probe::CallKDTree>();

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
PROBE_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
PROBE_API
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
        MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_COMP_REV, 0
#if defined(DEBUG) || defined(_DEBUG)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(MEGAMOL_CORE_DIRTY) && (MEGAMOL_CORE_DIRTY != 0)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
        );

    SetLibraryVersionInfo(ci->libs[1], "vislib",
        vislib::VISLIB_VERSION_MAJOR, vislib::VISLIB_VERSION_MINOR, vislib::VISLIB_VERSION_REVISION, 0
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
PROBE_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
PROBE_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
PROBE_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

/*
 * mmstd.moldyn.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmstd_moldyn/mmstd_moldyn.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "mmcore/factories/ModuleAutoDescription.h"
#include "mmcore/factories/LoaderADModuleAutoDescription.h"

#include "io/IMDAtomDataSource.h"
#include "io/MMSPDDataSource.h"
#include "io/SIFFDataSource.h"
#include "io/SIFFWriter.h"
#include "io/VIMDataSource.h"
#include "io/VisIttDataSource.h"
#include "io/VTFDataSource.h"
#include "io/VTFResDataSource.h"
#include "misc/ParticleWorker.h"
#include "io/XYZLoader.h"
#include "io/TclMolSelectionLoader.h"
#include "mmstd_moldyn/BrickStatsCall.h"
#include "io/BrickStatsDataSource.h"
#include "rendering/BrickStatsRenderer.h"


/*
 * mmplgPluginAPIVersion
 */
MMSTD_MOLDYN_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
MMSTD_MOLDYN_API
megamol::core::utility::plugins::PluginCompatibilityInfo *
mmplgGetPluginCompatibilityInfo(
megamol::core::utility::plugins::ErrorCallback onError) {
    // compatibility information with core and vislib
    using megamol::core::utility::plugins::PluginCompatibilityInfo;
    using megamol::core::utility::plugins::LibraryVersionInfo;
    using megamol::core::utility::plugins::SetLibraryVersionInfo;

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

    return ci;
}


/*
 * mmplgReleasePluginCompatibilityInfo
 */
MMSTD_MOLDYN_API void mmplgReleasePluginCompatibilityInfo(
        megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : megamol::core::utility::plugins::Plugin200Instance(
                /* machine-readable plugin assembly name */
                "mmstd_moldyn",
                /* human-readable plugin description */
                "MegaMol Plugins for Molecular Dynamics Data Visualization") {
            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {
            // register modules here:
            this->module_descriptions.RegisterDescription< ::megamol::core::factories::LoaderADModuleAutoDescription< ::megamol::stdplugin::moldyn::io::IMDAtomDataSource> >();
            this->module_descriptions.RegisterDescription< ::megamol::core::factories::LoaderADModuleAutoDescription< ::megamol::stdplugin::moldyn::io::MMSPDDataSource> >();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::SIFFDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::SIFFWriter>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VIMDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VisIttDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VTFDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::VTFResDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::misc::ParticleWorker>();
            this->module_descriptions.RegisterDescription< ::megamol::core::factories::LoaderADModuleAutoDescription< ::megamol::stdplugin::moldyn::io::XYZLoader> >();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::io::TclMolSelectionLoader>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::BrickStatsDataSource>();
            this->module_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::rendering::BrickStatsRenderer>();
            // register calls here:
            this->call_descriptions.RegisterAutoDescription< ::megamol::stdplugin::moldyn::BrickStatsCall>();
        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgGetPluginInstance
 */
MMSTD_MOLDYN_API
megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
MMSTD_MOLDYN_API void mmplgReleasePluginInstance(
        megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

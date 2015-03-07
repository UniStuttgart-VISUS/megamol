/*
 * mmstd_datatools.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmstd_datatools/mmstd_datatools.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "DataSetTimeRewriteModule.h"
#include "ParticleListMergeModule.h"
#include "DataFileSequenceStepper.h"
#include "SphereDataUnifier.h"
#include "ParticleThinner.h"
#include "OverrideParticleGlobals.h"
#include "ParticleRelaxationModule.h"
#include "ParticleListSelector.h"
#include "ParticleDensityOpacityModule.h"
#include "ForceCubicCBoxModule.h"
#include "DumpIColorHistogramModule.h"
#include "DataFileSequence.h"


/*
 * mmplgPluginAPIVersion
 */
MMSTD_DATATOOLS_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
MMSTD_DATATOOLS_API
megamol::core::utility::plugins::PluginCompatibilityInfo *
mmplgGetPluginCompatibilityInfo(
        megamol::core::utility::plugins::ErrorCallback onError) {
    // compatibility information with core and vislib
    using megamol::core::utility::plugins::PluginCompatibilityInfo;
    using megamol::core::utility::plugins::LibraryVersionInfo;

    PluginCompatibilityInfo *ci = new PluginCompatibilityInfo;
    ci->libs_cnt = 2;
    ci->libs = new LibraryVersionInfo[2];

    MEGAMOLCORE_PLUGIN200UTIL_Set_LibraryVersionInfo_V4(ci->libs[0], "MegaMolCore", MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_MAJOR_REV, MEGAMOL_CORE_MINOR_REV)
    MEGAMOLCORE_PLUGIN200UTIL_Set_LibraryVersionInfo_V4(ci->libs[1], "vislib", VISLIB_VERSION_MAJOR, VISLIB_VERSION_MINOR, VISLIB_VERSION_REVISION, VISLIB_VERSION_BUILD)

    return ci;
}


/*
 * mmplgReleasePluginCompatibilityInfo
 */
MMSTD_DATATOOLS_API void mmplgReleasePluginCompatibilityInfo(
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
                "mmstd_datatools",
                /* human-readable plugin description */
                "MegaMol Standard-Plugin containing data manipulation and conversion modules") {
            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {
            // register modules here:
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DataSetTimeRewriteModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleListMergeModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DataFileSequenceStepper>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::SphereDataUnifier>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleThinner>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::OverrideParticleGlobals>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleRelaxationModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleListSelector>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleDensityOpacityModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ForceCubicCBoxModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DumpIColorHistogramModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DataFileSequence>();
            // register calls here:
            // ...
        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgGetPluginInstance
 */
MMSTD_DATATOOLS_API
megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
MMSTD_DATATOOLS_API void mmplgReleasePluginInstance(
        megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

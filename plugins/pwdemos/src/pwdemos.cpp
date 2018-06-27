/*
 * pwdemos.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "pwdemos/pwdemos.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "AOSphereRenderer.h"
#include "BezierCPUMeshRenderer.h"
#include "QuartzCrystalDataSource.h"
#include "CrystalStructureVolumeRenderer.h"
#include "QuartzCrystalRenderer.h"
#include "QuartzDataGridder.h"
#include "QuartzRenderer.h"
#include "QuartzParticleFortLoader.h"
#include "QuartzTexRenderer.h"
#include "QuartzPlaneTexRenderer.h"
#include "QuartzPlaneRenderer.h"
#include "PoreNetExtractor.h"

#include "QuartzCrystalDataCall.h"
#include "QuartzParticleDataCall.h"
#include "QuartzParticleGridDataCall.h"


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "pwdemos", // TODO: Change this!

                /* human-readable plugin description */
                "Describing pwdemos (TODO: Change this!)") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

            this->module_descriptions.RegisterAutoDescription<megamol::demos::AOSphereRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::BezierCPUMeshRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::CrystalDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::CrystalRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzPlaneRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzPlaneTexRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::DataGridder>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::ParticleFortLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::QuartzTexRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::demos::PoreNetExtractor>();


#if (defined(WITH_CUDA) && (WITH_CUDA))
            this->module_descriptions.RegisterAutoDescription<megamol::demos::CrystalStructureVolumeRenderer>();
#endif

            // register calls here:

            this->call_descriptions.RegisterAutoDescription<megamol::demos::CrystalDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::demos::ParticleDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::demos::ParticleGridDataCall>();

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
PWDEMOS_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
PWDEMOS_API
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
PWDEMOS_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
PWDEMOS_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
PWDEMOS_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

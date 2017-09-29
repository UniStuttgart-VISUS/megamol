/*
 * OSPRay_plugin.cpp
 * Copyright (C) 2009-2017 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "OSPRay_plugin/OSPRay_plugin.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "OSPRayRenderer.h"

#include "OSPRaySphereGeometry.h"
#include "OSPRayNHSphereGeometry.h"
#include "OSPRayTriangleMesh.h"
#include "OSPRayStructuredVolume.h"
#include "OSPRayStreamLines.h"
#include "OSPRay_plugin/CallOSPRayStructure.h"

#include "OSPRayDistantLight.h"
#include "OSPRayAmbientLight.h"
#include "OSPRayPointLight.h"
#include "OSPRaySpotLight.h"
#include "OSPRayQuadLight.h"
#include "OSPRayHDRILight.h"
#include "CallOSPRayLight.h"

#include "OSPRayOBJMaterial.h"
#include "OSPRayLuminousMaterial.h"
#include "OSPRayVelvetMaterial.h"
#include "OSPRayMatteMaterial.h"
#include "OSPRayMetalMaterial.h"
#include "OSPRayMetallicPaintMaterial.h"
#include "OSPRayGlassMaterial.h"
#include "OSPRayThinGlassMaterial.h"
#include "OSPRayPlasticMaterial.h"
#include "OSPRay_plugin/CallOSPRayMaterial.h"



/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "OSPRay_plugin", 

                /* human-readable plugin description */
                "CPU Raytracing") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayRenderer>();

               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRaySphereGeometry>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayNHSphereGeometry>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayTriangleMesh>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayStructuredVolume>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayStreamLines>();

               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayDistantLight>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayAmbientLight>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayPointLight>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRaySpotLight>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayQuadLight>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayHDRILight>();

               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayOBJMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayLuminousMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayVelvetMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayMatteMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayMetalMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayMetallicPaintMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayGlassMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayThinGlassMaterial>();
               this->module_descriptions.RegisterAutoDescription<megamol::ospray::OSPRayPlasticMaterial>();


            // register calls here:

               this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayLight>();
               this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayStructure>();
               this->call_descriptions.RegisterAutoDescription<megamol::ospray::CallOSPRayMaterial>();

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
OSPRAY_PLUGIN_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
OSPRAY_PLUGIN_API
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
        VISLIB_VERSION_MAJOR, VISLIB_VERSION_MINOR, VISLIB_VERSION_REVISION, 0
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
OSPRAY_PLUGIN_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
OSPRAY_PLUGIN_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
OSPRAY_PLUGIN_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

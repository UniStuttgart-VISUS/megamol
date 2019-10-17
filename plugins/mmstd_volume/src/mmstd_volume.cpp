/*
 * mmstd_volume.cpp
 *
 * Copyright (C) 2009-2019 by Universitaet Stuttgart (VISUS).
 * All rights reserved.
 */

#include "stdafx.h"
#include "mmstd_volume/mmstd_volume.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "BuckyBall.h"
#include "DatRawWriter.h"
#include "RaycastVolumeRenderer.h"
#include "VolumeSliceRenderer.h"
#include "VolumetricDataSource.h"

/* anonymous namespace hides this type from any other object file */
namespace {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "mmstd_volume",

              /* human-readable plugin description */
              "Provides modules for volume rendering"){
          };

    /** Dtor */
    virtual ~plugin_instance(void) {
    }

    /** Registers modules and calls */
    virtual void registerClasses(void) {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::BuckyBall>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::DatRawWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::RaycastVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::VolumeSliceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::volume::VolumetricDataSource>();

        // register calls here:

    }

    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
};
} // namespace


/*
 * mmplgPluginAPIVersion
 */
MMSTD_VOLUME_API int mmplgPluginAPIVersion(void){ MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion }


/*
 * mmplgGetPluginCompatibilityInfo
 */
MMSTD_VOLUME_API ::megamol::core::utility::plugins::PluginCompatibilityInfo* mmplgGetPluginCompatibilityInfo(
    ::megamol::core::utility::plugins::ErrorCallback onError) {
    // compatibility information with core and vislib
    using ::megamol::core::utility::plugins::LibraryVersionInfo;
    using ::megamol::core::utility::plugins::PluginCompatibilityInfo;

    PluginCompatibilityInfo* ci = new PluginCompatibilityInfo;
    ci->libs_cnt = 2;
    ci->libs = new LibraryVersionInfo[2];

    SetLibraryVersionInfo(
        ci->libs[0], "MegaMolCore", MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_COMP_REV,
        0
#if defined(DEBUG) || defined(_DEBUG)
            | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(MEGAMOL_CORE_DIRTY) && (MEGAMOL_CORE_DIRTY != 0)
            | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
    );

    SetLibraryVersionInfo(ci->libs[1], "vislib", vislib::VISLIB_VERSION_MAJOR, vislib::VISLIB_VERSION_MINOR,
        vislib::VISLIB_VERSION_REVISION,
        0
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
MMSTD_VOLUME_API
void mmplgReleasePluginCompatibilityInfo(::megamol::core::utility::plugins::PluginCompatibilityInfo* ci){
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)}


/*
 * mmplgGetPluginInstance
 */
MMSTD_VOLUME_API ::megamol::core::utility::plugins::AbstractPluginInstance* mmplgGetPluginInstance(
    ::megamol::core::utility::plugins::ErrorCallback onError){
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)}


/*
 * mmplgReleasePluginInstance
 */
MMSTD_VOLUME_API void mmplgReleasePluginInstance(::megamol::core::utility::plugins::AbstractPluginInstance* pi){
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)}

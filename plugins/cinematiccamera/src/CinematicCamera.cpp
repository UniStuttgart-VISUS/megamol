/*
 * CinematicCamera.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "CinematicCamera/CinematicCamera.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "CinematicRenderer.h"
#include "TimeLineRenderer.h"
#include "CallCinematicCamera.h"
#include "KeyframeKeeper.h"
#include "PreviewView.h"
#include "CinematicView.h"

/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:

        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "CinematicCamera", 

                /* human-readable plugin description */
                "Describing CinematicCamera (TODO: Change this!)") {
			
            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {
			
            // register modules here:
			this->module_descriptions.RegisterAutoDescription<megamol::cinematiccamera::CinematicRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::cinematiccamera::TimeLineRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::cinematiccamera::KeyframeKeeper>();
			this->module_descriptions.RegisterAutoDescription<megamol::cinematiccamera::PreviewView>();
			this->module_descriptions.RegisterAutoDescription<megamol::cinematiccamera::CinematicView>();
            //
            // TODO: Register your plugin's modules here
            // like:
            //   this->module_descriptions.RegisterAutoDescription<megamol::CinematicCamera::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::CinematicCamera::MyModule2>();
            //   ...
            //

            // register calls here:
			this->call_descriptions.RegisterAutoDescription < megamol::cinematiccamera::CallCinematicCamera>();
            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::CinematicCamera::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::CinematicCamera::MyCall2>();
            //   ...
            //

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
CINEMATICCAMERA_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
CINEMATICCAMERA_API
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
CINEMATICCAMERA_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
CINEMATICCAMERA_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
CINEMATICCAMERA_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

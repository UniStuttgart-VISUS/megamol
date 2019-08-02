/*
 * archvis.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "archvis/archvis.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "ArchVisMSMDataSource.h"
#include "CreateFEMModel.h"
#include "CreateMSM.h"
#include "FEMDataCall.h"
#include "FEMModel.h"
#include "FEMMeshDataSource.h"
#include "FEMMaterialDataSource.h"
#include "FEMRenderTaskDataSource.h"
#include "FEMLoader.h"
#include "MSMConvexHullMeshDataSource.h"
#include "MSMDataCall.h"
#include "MSMRenderTaskDataSource.h"


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "archvis", // TODO: Change this!

                /* human-readable plugin description */
                "Describing archvis (TODO: Change this!)") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:

            //
            // TODO: Register your plugin's modules here
            // like:
            //   this->module_descriptions.RegisterAutoDescription<megamol::archvis::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::archvis::MyModule2>();
            //   ...
            //
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::ArchVisMSMDataSource>();

            this->module_descriptions.RegisterAutoDescription<megamol::archvis::CreateFEMModel>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::CreateMSM>();
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMMeshDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMMaterialDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMRenderTaskDataSource>();
			this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::MSMConvexHullDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::MSMRenderTaskDataSource>();


            // register calls here:

            //
            // TODO: Register your plugin's calls here
            // like:
            //   this->call_descriptions.RegisterAutoDescription<megamol::archvis::MyCall1>();
            //   this->call_descriptions.RegisterAutoDescription<megamol::archvis::MyCall2>();
            //   ...
            //
			this->call_descriptions.RegisterAutoDescription<megamol::archvis::FEMDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::archvis::MSMDataCall>();

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
ARCHVIS_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
ARCHVIS_API
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
ARCHVIS_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
ARCHVIS_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
ARCHVIS_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

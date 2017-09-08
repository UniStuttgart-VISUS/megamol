/*
 * beztube.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "beztube/beztube.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "ext/ExtBezierDataCall.h"
#include "ext/ExtBezierDataSource.h"
#include "ext/ExtBezierMeshRenderer.h"
#include "ext/ExtBezierRaycastRenderer.h"
#include "BezierControlLines.h"
#include "v1/BezierDataCall.h"
#include "v1/BezierDataSource.h"
#include "v1/BezierMeshRenderer.h"
#include "v1/BezierRaycastRenderer.h"
#include "BezDatMigrate.h"
#include "BezDatWriter.h"
#include "BezDatReader.h"
#include "mmcore/factories/LoaderADModuleAutoDescription.h"
#include "BezDatOpt.h"
#include "BezierLines.h"
#include "BezierCPUMeshRenderer.h"
#include "salm/BezierTessRenderer.h"


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "beztube",

                /* human-readable plugin description */
                "Bezier Tube rendering plugin") {

            // here we could perform addition initialization
        };
        /** Dtor */
        virtual ~plugin_instance(void) {
            // here we could perform addition de-initialization
        }
        /** Registers modules and calls */
        virtual void registerClasses(void) {

            // register modules here:
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::ext::ExtBezierDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::ext::ExtBezierMeshRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::ext::ExtBezierRaycastRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::BezierControlLines>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::v1::BezierDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::v1::BezierMeshRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::v1::BezierRaycastRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::BezDatMigrate>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::BezDatWriter>();
            this->module_descriptions.RegisterDescription<megamol::core::factories::LoaderADModuleAutoDescription<megamol::beztube::BezDatReader> >();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::BezDatOpt>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::BezierLines>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::BezierCPUMeshRenderer>();
            this->module_descriptions.RegisterAutoDescription<megamol::beztube::salm::BezierTessRenderer>();

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::beztube::ext::ExtBezierDataCall>();
            this->call_descriptions.RegisterDescription<megamol::beztube::v1::BezierDataCallDescription>();

        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
BEZTUBE_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
BEZTUBE_API
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
BEZTUBE_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
BEZTUBE_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
BEZTUBE_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

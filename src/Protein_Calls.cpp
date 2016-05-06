/*
 * Protein_Calls.cpp
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "protein_calls/Protein_Calls.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "protein_calls/BindingSiteCall.h"
#include "protein_calls/CrystalStructureDataCall.h"
#include "protein_calls/DiagramCall.h"
#include "protein_calls/IntSelectionCall.h"
#include "protein_calls/MolecularDataCall.h"
#include "protein_calls/ResidueSelectionCall.h"
#include "protein_calls/SplitMergeCall.h"
#include "protein_calls/VTIDataCall.h"


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : ::megamol::core::utility::plugins::Plugin200Instance(

                /* machine-readable plugin assembly name */
                "Protein_Calls", // TODO: Change this!

                /* human-readable plugin description */
                "Plugin containing calls used by Protein and Protein_CUDA") {

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
            //   this->module_descriptions.RegisterAutoDescription<megamol::Protein_Calls::MyModule1>();
            //   this->module_descriptions.RegisterAutoDescription<megamol::Protein_Calls::MyModule2>();
            //   ...
            //

            // register calls here:
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::BindingSiteCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::CrystalStructureDataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::DiagramCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::IntSelectionCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::MolecularDataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::ResidueSelectionCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::SplitMergeCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_calls::VTIDataCall>();
        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgPluginAPIVersion
 */
PROTEIN_CALLS_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
PROTEIN_CALLS_API
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
PROTEIN_CALLS_API
void mmplgReleasePluginCompatibilityInfo(
        ::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
 * mmplgGetPluginInstance
 */
PROTEIN_CALLS_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        ::megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
PROTEIN_CALLS_API
void mmplgReleasePluginInstance(
        ::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

/*
 * Protein_CUDA.cpp
 *
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "../include/protein_cuda/Protein_CUDA.h"
#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/versioninfo.h"

// jobs
#include "DataWriter.h"

// 3D renderers
#include "MoleculeCudaSESRenderer.h"
#include "MoleculeCBCudaRenderer.h"
#include "CrystalStructureVolumeRenderer.h"
#include "QuickSESRenderer.h"
#include "QuickSurfRenderer.h"
#include "QuickSurfRenderer2.h"
#include "QuickSurfMTRenderer.h"
#include "QuickSurfRaycaster.h"
#include "MoleculeVolumeCudaRenderer.h"
#include "VolumeMeshRenderer.h"
#include "ComparativeFieldTopologyRenderer.h"
#include "PotentialVolumeRaycaster.h"
#include "SurfacePotentialRendererSlave.h"
#include "StreamlineRenderer.h"
#include "ComparativeMolSurfaceRenderer.h"

// 2D renderers
#include "SecStructRenderer2D.h"

// data sources
#include "Filter.h"

// data interfaces (calls)
#include "mmcore/CallVolumeData.h"
#include "VBODataCall.h"
#include "PlaneDataCall.h"

// other modules (filter etc)
#include "PotentialCalculator.h"
#include "ProteinVariantMatch.h"
#include "SecStructFlattener.h"
#include "ParticlesToMeshConverter.h"

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "vislib/vislibversion.h"

#include "vislib/sys/Log.h"
#include "vislib/Trace.h"

/* anonymous namespace hides this type from any other object files */
namespace {
	/** Implementing the instance class of this plugin */
	class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
	public:
		/** ctor */
		plugin_instance(void)
			: ::megamol::core::utility::plugins::Plugin200Instance(

			/* machine-readable plugin assembly name */
			"Protein_CUDA",

			/* human-readable plugin description */
			"Plugin for protein rendering using CUDA for accelleration (SFB716 D4)") {

				// here we could perform addition initialization
			};
		/** Dtor */
		virtual ~plugin_instance(void) {
			// here we could perform addition de-initialization
		}
		/** Registers modules and calls */
		virtual void registerClasses(void) {

			// register modules here:
#ifdef WITH_OPENHAPTICS
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::HapticsMoleculeRenderer>();
#endif // WITH_OPENHAPTICS
#ifdef WITH_OPENBABEL
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::OpenBabelLoader>();
#endif // WITH_OPENBABEL
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::PotentialVolumeRaycaster>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::StreamlineRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::MoleculeCudaSESRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::MoleculeCBCudaRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::QuickSESRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::QuickSurfRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::QuickSurfRenderer2>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::QuickSurfMTRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::MoleculeVolumeCudaRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::VolumeMeshRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::DataWriter>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::CrystalStructureVolumeRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::ComparativeMolSurfaceRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::ComparativeFieldTopologyRenderer>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::ProteinVariantMatch>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::QuickSurfRaycaster>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SecStructFlattener>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::ParticlesToMeshConverter>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SecStructRenderer2D>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::PotentialCalculator>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::Filter>();
			this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SurfacePotentialRendererSlave>();

			// register calls here:
			this->call_descriptions.RegisterAutoDescription<megamol::protein_cuda::VBODataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_cuda::PlaneDataCall>();
		}
		MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
	};
}

/*
 * mmplgPluginAPIVersion
 */
PROTEIN_CUDA_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
PROTEIN_CUDA_API
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
PROTEIN_CUDA_API
void mmplgReleasePluginCompatibilityInfo(
::megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
	// release compatiblity data on the correct heap
	MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/*
* mmplgGetPluginInstance
*/
PROTEIN_CUDA_API
::megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
::megamol::core::utility::plugins::ErrorCallback onError) {
	MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
* mmplgReleasePluginInstance
*/
PROTEIN_CUDA_API
void mmplgReleasePluginInstance(
::megamol::core::utility::plugins::AbstractPluginInstance* pi) {
	MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}
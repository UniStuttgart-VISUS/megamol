/*
 * Protein_CUDA.cpp
 *
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/PluginRegister.h"
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
#include "SombreroWarper.h"

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "vislib/vislibversion.h"

#include "vislib/Trace.h"

namespace megamol::protein_cuda {
	/** Implementing the instance class of this plugin */
	class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
        REGISTERPLUGIN(plugin_instance)
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
            this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SombreroWarper>();

			// register calls here:
			this->call_descriptions.RegisterAutoDescription<megamol::protein_cuda::VBODataCall>();
			this->call_descriptions.RegisterAutoDescription<megamol::protein_cuda::PlaneDataCall>();
		}
	};
} // namespace megamol::protein_cuda

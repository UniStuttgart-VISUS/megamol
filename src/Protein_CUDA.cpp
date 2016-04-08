/*
 * Protein.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "../include/protein_cuda/Protein_CUDA.h"
#include "mmcore/api/MegaMolCore.std.h"

// views
//#include "View3DSpaceMouse.h"
//#include "View3DMouse.h"

// jobs
#include "DataWriter.h"
//#include "PDBWriter.h"
//#include "VTIWriter.h"

// 3D renderers
//#include "ProteinVolumeRenderer.h"
//#include "SolventVolumeRenderer.h"
//#include "SimpleMoleculeRenderer.h"
//#include "SphereRenderer.h"
//#include "SolPathRenderer.h"
//#include "MoleculeSESRenderer.h"
//#include "MoleculeCartoonRenderer.h"
#include "MoleculeCudaSESRenderer.h"
//#include "ElectrostaticsRenderer.h"
#include "MoleculeCBCudaRenderer.h"
//#include "HapticsMoleculeRenderer.h"
//#include "SSAORendererDeferred.h"
//#include "ToonRendererDeferred.h"
#include "CrystalStructureVolumeRenderer.h"
//#include "DofRendererDeferred.h"
//#include "SphereRendererMouse.h"
#include "QuickSurfRenderer.h"
#include "QuickSurfRenderer2.h"
#include "QuickSurfMTRenderer.h"
#include "MoleculeVolumeCudaRenderer.h"
//#include "GLSLVolumeRenderer.h"
#include "VolumeMeshRenderer.h"
#include "ComparativeFieldTopologyRenderer.h"
#include "PotentialVolumeRaycaster.h"
#include "SurfacePotentialRendererSlave.h"
#include "StreamlineRenderer.h"
//#include "VariantMatchRenderer.h"
//#include "SecPlaneRenderer.h"
#include "ComparativeMolSurfaceRenderer.h"
//#include "UnstructuredGridRenderer.h"
//#include "VolumeDirectionRenderer.h"
//#include "LayeredIsosurfaceRenderer.h"

// 2D renderers
//#include "VolumeSliceRenderer.h"
//#include "Diagram2DRenderer.h"
//#include "DiagramRenderer.h"
//#include "SplitMergeRenderer.h"
//#include "SequenceRenderer.h"

// data sources
//#include "PDBLoader.h"
//#include "SolPathDataSource.h"
//#include "CCP4VolumeData.h"
//#include "CoarseGrainDataLoader.h"
//#include "FrodockLoader.h"
//#include "XYZLoader.h"
#include "Filter.h"
//#include "SolventDataGenerator.h"
//#include "GROLoader.h"
//#include "CrystalStructureDataSource.h"
//#include "VTILoader.h"
//#include "VMDDXLoader.h"
//#include "TrajectorySmoothFilter.h"
//#include "BindingSiteDataSource.h"
//#include "AggregatedDensity.h"
//#include "ResidueSelection.h"
//#include "VTKLegacyDataLoaderUnstructuredGrid.h"
//#include "MolecularBezierData.h"
//#include "MultiPDBLoader.h"
//#include "OpenBabelLoader.h"

// data interfaces (calls)
//#include "SolPathDataCall.h"
#include "mmcore/CallVolumeData.h"
//#include "CallColor.h"
//#include "SphereDataCall.h"
//#include "VolumeSliceCall.h"
//#include "Diagram2DCall.h"
//#include "ParticleDataCall.h"
//#include "ForceDataCall.h"
//#include "CrystalStructureDataCall.h"
//#include "CallMouseInput.h"
//#include "DiagramCall.h"
//#include "SplitMergeCall.h"
//#include "IntSelectionCall.h"
//#include "ResidueSelectionCall.h"
//#include "VTIDataCall.h"
#include "VBODataCall.h"
#include "VariantMatchDataCall.h"
//#include "VTKLegacyDataCallUnstructuredGrid.h"


//#include "MoleculeBallifier.h"

// other modules (filter etc)
//#include "ColorModule.h"
//#include "IntSelection.h"
#include "PotentialCalculator.h"
#include "ProteinVariantMatch.h"
//#include "MultiParticleDataFilter.h"
//#include "PDBInterpolator.h"

#include "mmcore/factories/CallAutoDescription.h"
#include "mmcore/factories/ModuleAutoDescription.h"
#include "vislib/vislibversion.h"

#include "vislib/sys/Log.h"
#include "vislib/Trace.h"
#include "vislib/sys/ThreadSafeStackTrace.h"



/*
 * mmplgPluginAPIVersion
 */
PROTEIN_API int mmplgPluginAPIVersion(void) {
    return 100;
}


/*
 * mmplgPluginName
 */
PROTEIN_API const char * mmplgPluginName(void) {
    return "Protein_CUDA";
}


/*
 * mmplgPluginDescription
 */
PROTEIN_API const char * mmplgPluginDescription(void) {
    return "Plugin for protein rendering using CUDA for accelleration (SFB716 D4)";
}


/*
 * mmplgCoreCompatibilityValue
 */
PROTEIN_API const void * mmplgCoreCompatibilityValue(void) {
    static const mmplgCompatibilityValues compRev = {
        sizeof(mmplgCompatibilityValues),
        MEGAMOL_CORE_COMP_REV,
        VISLIB_VERSION_REVISION
    };
    return &compRev;
}


/*
 * mmplgModuleCount
 */
PROTEIN_API int mmplgModuleCount(void) {
    //int moduleCount = 53;
	int moduleCount = 3;
#ifdef WITH_CUDA
    moduleCount+=14;
#endif // WITH_CUDA
#ifdef WITH_OPENHAPTICS
    moduleCount+=1;
#endif // WITH_OPENHAPTICS
#ifdef WITH_OPENBABEL
    moduleCount += 1;
#endif // WITH_OPENBABEL
    return moduleCount;
}


/*
 * mmplgModuleDescription
 */
PROTEIN_API void* mmplgModuleDescription(int idx) {
    using namespace megamol;
    using namespace megamol::core;
    switch (idx) {
#ifdef WITH_OPENHAPTICS
        case 0 : return new megamol::core::ModuleAutoDescription<megamol::protein_cuda::HapticsMoleculeRenderer>();
        #define HAPTICS_OFFSET 1
#else
        #define HAPTICS_OFFSET 0
#endif // WITH_OPENHAPTICS
#ifdef WITH_CUDA
        case HAPTICS_OFFSET +  0 : return new factories::ModuleAutoDescription<protein_cuda::PotentialVolumeRaycaster>();
        case HAPTICS_OFFSET +  1 : return new factories::ModuleAutoDescription<protein_cuda::StreamlineRenderer>();
        case HAPTICS_OFFSET +  2 : return new factories::ModuleAutoDescription<protein_cuda::MoleculeCudaSESRenderer>();
        case HAPTICS_OFFSET +  3 : return new factories::ModuleAutoDescription<protein_cuda::MoleculeCBCudaRenderer>();
        case HAPTICS_OFFSET +  4 : return new factories::ModuleAutoDescription<protein_cuda::QuickSurfRenderer>();
        case HAPTICS_OFFSET +  5 : return new factories::ModuleAutoDescription<protein_cuda::QuickSurfRenderer2>();
        case HAPTICS_OFFSET +  6 : return new factories::ModuleAutoDescription<protein_cuda::QuickSurfMTRenderer>();
        case HAPTICS_OFFSET +  7 : return new factories::ModuleAutoDescription<protein_cuda::MoleculeVolumeCudaRenderer>();
        case HAPTICS_OFFSET +  8 : return new factories::ModuleAutoDescription<protein_cuda::VolumeMeshRenderer>();
        case HAPTICS_OFFSET +  9 : return new factories::ModuleAutoDescription<protein_cuda::DataWriter>();
        case HAPTICS_OFFSET + 10 : return new factories::ModuleAutoDescription<protein_cuda::CrystalStructureVolumeRenderer>();
        case HAPTICS_OFFSET + 11 : return new factories::ModuleAutoDescription<protein_cuda::ComparativeMolSurfaceRenderer>();
        case HAPTICS_OFFSET + 12 : return new factories::ModuleAutoDescription<protein_cuda::ComparativeFieldTopologyRenderer>();
        case HAPTICS_OFFSET + 13 : return new factories::ModuleAutoDescription<protein_cuda::ProteinVariantMatch>();
        #define CUDA_OFFSET 14
#else
        #define CUDA_OFFSET 0
#endif // WITH_CUDA
#ifdef WITH_OPENBABEL
		case CUDA_OFFSET + HAPTICS_OFFSET + 0: return new factories::ModuleAutoDescription<protein_cuda::OpenBabelLoader>();
        #define OPENBABAEL_OFFSET 1
#else
        #define OPENBABAEL_OFFSET 0
#endif //WITH_OPENBABEL
		case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 0: return new factories::ModuleAutoDescription<protein_cuda::PotentialCalculator>();
		case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 1: return new factories::ModuleAutoDescription<protein_cuda::Filter>();
		case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 2: return new factories::ModuleAutoDescription<protein_cuda::SurfacePotentialRendererSlave>();
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgCallCount
 */
PROTEIN_API int mmplgCallCount(void) {
    //return 18;
	return 2;
}


/*
 * mmplgCallDescription
 */
PROTEIN_API void* mmplgCallDescription(int idx) {
    switch (idx) {
		case 0: return new megamol::core::factories::CallAutoDescription<megamol::protein_cuda::VariantMatchDataCall>();
		case 1: return new megamol::core::factories::CallAutoDescription<megamol::protein_cuda::VBODataCall>();
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgConnectStatics
 */
PROTEIN_API bool mmplgConnectStatics(int which, void* value) {
#if defined(DEBUG) || defined(_DEBUG)
    // only trace non-vislib messages
    vislib::Trace::GetInstance().SetLevel(vislib::Trace::LEVEL_VL);
#endif /* DEBUG || _DEBUG */

    switch (which) {
        case 1: // vislib::log
            vislib::sys::Log::DefaultLog.SetLogFileName(static_cast<const char*>(NULL), false);
            vislib::sys::Log::DefaultLog.SetLevel(vislib::sys::Log::LEVEL_NONE);
            vislib::sys::Log::DefaultLog.SetEchoTarget(new
                vislib::sys::Log::RedirectTarget(static_cast<vislib::sys::Log*>(value)));
            vislib::sys::Log::DefaultLog.SetEchoLevel(vislib::sys::Log::LEVEL_ALL);
            vislib::sys::Log::DefaultLog.EchoOfflineMessages(true);
            return true;
        case 2: // vislib::stacktrace
            return vislib::sys::ThreadSafeStackTrace::Initialise(
                *static_cast<const vislib::SmartPtr<vislib::StackTrace>*>(value), true);
    }
    return false;
}

/*
 * Protein.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "Protein.h"
#include "mmcore/api/MegaMolCore.std.h"

// views
#include "View3DSpaceMouse.h"
#include "View3DMouse.h"

// jobs
#include "DataWriter.h"
#include "PDBWriter.h"
#include "VTIWriter.h"

// 3D renderers
#include "ProteinVolumeRenderer.h"
#include "SolventVolumeRenderer.h"
#include "SimpleMoleculeRenderer.h"
#include "SphereRenderer.h"
#include "SolPathRenderer.h"
#include "MoleculeSESRenderer.h"
#include "MoleculeCartoonRenderer.h"
#include "MoleculeCudaSESRenderer.h"
#include "ElectrostaticsRenderer.h"
#include "MoleculeCBCudaRenderer.h"
#include "HapticsMoleculeRenderer.h"
#include "SSAORendererDeferred.h"
#include "ToonRendererDeferred.h"
#include "CrystalStructureVolumeRenderer.h"
#include "DofRendererDeferred.h"
#include "SphereRendererMouse.h"
#include "QuickSurfRenderer.h"
#include "QuickSurfRenderer2.h"
#include "QuickSurfMTRenderer.h"
#include "MoleculeVolumeCudaRenderer.h"
#include "GLSLVolumeRenderer.h"
#include "VolumeMeshRenderer.h"
#include "ComparativeFieldTopologyRenderer.h"
#include "PotentialVolumeRaycaster.h"
#include "SurfacePotentialRendererSlave.h"
#include "StreamlineRenderer.h"
#include "VariantMatchRenderer.h"
#include "SecPlaneRenderer.h"
#include "ComparativeMolSurfaceRenderer.h"
#include "UnstructuredGridRenderer.h"
#include "VolumeDirectionRenderer.h"
#include "LayeredIsosurfaceRenderer.h"

// 2D renderers
#include "VolumeSliceRenderer.h"
#include "Diagram2DRenderer.h"
#include "DiagramRenderer.h"
#include "SplitMergeRenderer.h"
#include "SequenceRenderer.h"

// data sources
#include "PDBLoader.h"
#include "SolPathDataSource.h"
#include "CCP4VolumeData.h"
#include "CoarseGrainDataLoader.h"
#include "FrodockLoader.h"
#include "XYZLoader.h"
#include "Filter.h"
#include "SolventDataGenerator.h"
#include "GROLoader.h"
#include "CrystalStructureDataSource.h"
#include "VTILoader.h"
#include "VMDDXLoader.h"
#include "TrajectorySmoothFilter.h"
#include "BindingSiteDataSource.h"
#include "AggregatedDensity.h"
#include "ResidueSelection.h"
#include "VTKLegacyDataLoaderUnstructuredGrid.h"
#include "MolecularBezierData.h"
#include "MultiPDBLoader.h"
#include "OpenBabelLoader.h"

// data interfaces (calls)
#include "SolPathDataCall.h"
#include "mmcore/CallVolumeData.h"
#include "CallColor.h"
#include "SphereDataCall.h"
#include "VolumeSliceCall.h"
#include "Diagram2DCall.h"
#include "ParticleDataCall.h"
#include "ForceDataCall.h"
#include "CrystalStructureDataCall.h"
#include "CallMouseInput.h"
#include "DiagramCall.h"
#include "SplitMergeCall.h"
#include "IntSelectionCall.h"
#include "ResidueSelectionCall.h"
#include "VTIDataCall.h"
#include "VBODataCall.h"
#include "VariantMatchDataCall.h"
#include "VTKLegacyDataCallUnstructuredGrid.h"


#include "MoleculeBallifier.h"

// other modules (filter etc)
#include "ColorModule.h"
#include "IntSelection.h"
#include "PotentialCalculator.h"
#include "ProteinVariantMatch.h"
#include "MultiParticleDataFilter.h"
#include "PDBInterpolator.h"

#include "mmcore/CallAutoDescription.h"
#include "mmcore/ModuleAutoDescription.h"
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
    return "Protein";
}


/*
 * mmplgPluginDescription
 */
PROTEIN_API const char * mmplgPluginDescription(void) {
    return "Plugin for protein rendering (SFB716 D4)";
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
    int moduleCount = 53;
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
        case 0 : return new megamol::core::ModuleAutoDescription<megamol::protein::HapticsMoleculeRenderer>();
        #define HAPTICS_OFFSET 1
#else
        #define HAPTICS_OFFSET 0
#endif // WITH_OPENHAPTICS
#ifdef WITH_CUDA
        case HAPTICS_OFFSET +  0 : return new ModuleAutoDescription<protein::PotentialVolumeRaycaster>();
        case HAPTICS_OFFSET +  1 : return new ModuleAutoDescription<protein::StreamlineRenderer>();
        case HAPTICS_OFFSET +  2 : return new ModuleAutoDescription<protein::MoleculeCudaSESRenderer>();
        case HAPTICS_OFFSET +  3 : return new ModuleAutoDescription<protein::MoleculeCBCudaRenderer>();
        case HAPTICS_OFFSET +  4 : return new ModuleAutoDescription<protein::QuickSurfRenderer>();
        case HAPTICS_OFFSET +  5 : return new ModuleAutoDescription<protein::QuickSurfRenderer2>();
        case HAPTICS_OFFSET +  6 : return new ModuleAutoDescription<protein::QuickSurfMTRenderer>();
        case HAPTICS_OFFSET +  7 : return new ModuleAutoDescription<protein::MoleculeVolumeCudaRenderer>();
        case HAPTICS_OFFSET +  8 : return new ModuleAutoDescription<protein::VolumeMeshRenderer>();
        case HAPTICS_OFFSET +  9 : return new ModuleAutoDescription<protein::DataWriter>();
        case HAPTICS_OFFSET + 10 : return new ModuleAutoDescription<protein::CrystalStructureVolumeRenderer>();
        case HAPTICS_OFFSET + 11 : return new ModuleAutoDescription<protein::ComparativeMolSurfaceRenderer>();
        case HAPTICS_OFFSET + 12 : return new ModuleAutoDescription<protein::ComparativeFieldTopologyRenderer>();
        case HAPTICS_OFFSET + 13 : return new ModuleAutoDescription<protein::ProteinVariantMatch>();
        #define CUDA_OFFSET 14
#else
        #define CUDA_OFFSET 0
#endif // WITH_CUDA
#ifdef WITH_OPENBABEL
		case CUDA_OFFSET + HAPTICS_OFFSET + 0: return new ModuleAutoDescription<protein::OpenBabelLoader>();
		#define OPENBABAEL_OFFSET 1
#else
		#define OPENBABAEL_OFFSET 0
#endif //WITH_OPENBABEL
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  0 : return new ModuleAutoDescription<protein::SequenceRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  1 : return new ModuleAutoDescription<protein::BindingSiteDataSource>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  2 : return new ModuleAutoDescription<protein::SolPathDataSource>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  3 : return new ModuleAutoDescription<protein::SolPathRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  4 : return new ModuleAutoDescription<protein::ProteinVolumeRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  5 : return new ModuleAutoDescription<protein::CCP4VolumeData>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  6 : return new ModuleAutoDescription<protein::PDBLoader>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  7 : return new ModuleAutoDescription<protein::SimpleMoleculeRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  8 : return new ModuleAutoDescription<protein::CoarseGrainDataLoader>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET +  9 : return new ModuleAutoDescription<protein::SphereRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 10 : return new ModuleAutoDescription<protein::MoleculeSESRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 11 : return new ModuleAutoDescription<protein::MoleculeCartoonRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 12 : return new ModuleAutoDescription<protein::FrodockLoader>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 13 : return new ModuleAutoDescription<protein::VolumeSliceRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 14 : return new ModuleAutoDescription<protein::Diagram2DRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 15 : return new ModuleAutoDescription<protein::XYZLoader>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 16 : return new ModuleAutoDescription<protein::ElectrostaticsRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 17 : return new ModuleAutoDescription<protein::SolventVolumeRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 18 : return new ModuleAutoDescription<protein::Filter>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 19 : return new ModuleAutoDescription<protein::SolventDataGenerator>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 20 : return new ModuleAutoDescription<protein::View3DSpaceMouse>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 21 : return new ModuleAutoDescription<protein::GROLoader>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 22 : return new ModuleAutoDescription<protein::SSAORendererDeferred>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 23 : return new ModuleAutoDescription<protein::ToonRendererDeferred>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 24 : return new ModuleAutoDescription<protein::DofRendererDeferred>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 25 : return new ModuleAutoDescription<protein::SphereRendererMouse>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 26 : return new ModuleAutoDescription<protein::View3DMouse>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 27 : return new ModuleAutoDescription<protein::GLSLVolumeRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 28 : return new ModuleAutoDescription<protein::DiagramRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 29 : return new ModuleAutoDescription<protein::SplitMergeRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 30 : return new ModuleAutoDescription<protein::IntSelection>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 31 : return new ModuleAutoDescription<protein::CrystalStructureDataSource>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 32 : return new ModuleAutoDescription<protein::VTILoader>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 33 : return new ModuleAutoDescription<protein::PDBWriter>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 34 : return new ModuleAutoDescription<protein::VTIWriter>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 35 : return new ModuleAutoDescription<protein::PotentialCalculator>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 36 : return new ModuleAutoDescription<protein::VMDDXLoader>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 37 : return new ModuleAutoDescription<protein::TrajectorySmoothFilter>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 38 : return new ModuleAutoDescription<protein::SurfacePotentialRendererSlave>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 39 : return new ModuleAutoDescription<protein::VariantMatchRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 40 : return new ModuleAutoDescription<protein::MoleculeBallifier>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 41 : return new ModuleAutoDescription<protein::ResidueSelection>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 42 : return new ModuleAutoDescription<protein::SecPlaneRenderer>();
		case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 43 : return new ModuleAutoDescription<protein::AggregatedDensity>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 44 : return new ModuleAutoDescription<protein::VTKLegacyDataLoaderUnstructuredGrid>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 45 : return new ModuleAutoDescription<protein::UnstructuredGridRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 46 : return new ModuleAutoDescription<protein::MolecularBezierData>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 47 : return new ModuleAutoDescription<protein::MultiParticleDataFilter>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 48 : return new ModuleAutoDescription<protein::VolumeDirectionRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 49 : return new ModuleAutoDescription<protein::LayeredIsosurfaceRenderer>();
        case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 50 : return new ModuleAutoDescription<protein::MultiPDBLoader>();
		case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 51: return new ModuleAutoDescription<protein::ColorModule>();
		case OPENBABAEL_OFFSET + CUDA_OFFSET + HAPTICS_OFFSET + 52: return new ModuleAutoDescription<protein::PDBInterpolator>();
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgCallCount
 */
PROTEIN_API int mmplgCallCount(void) {
    return 18;
}


/*
 * mmplgCallDescription
 */
PROTEIN_API void* mmplgCallDescription(int idx) {
    switch (idx) {
        case 0: return new megamol::core::CallAutoDescription<megamol::protein::SolPathDataCall>();
        case 1: return new megamol::core::CallAutoDescription<megamol::protein::CallProteinVolumeData>();
        case 2: return new megamol::core::CallAutoDescription<megamol::protein::SphereDataCall>();
        case 3: return new megamol::core::CallAutoDescription<megamol::protein::VolumeSliceCall>();
        case 4: return new megamol::core::CallAutoDescription<megamol::protein::Diagram2DCall>();
        case 5: return new megamol::core::CallAutoDescription<megamol::protein::ParticleDataCall>();
        case 6: return new megamol::core::CallAutoDescription<megamol::protein::ForceDataCall>();
        case 7: return new megamol::core::CallAutoDescription<megamol::protein::CallMouseInput>();
        case 8: return new megamol::core::CallAutoDescription<megamol::protein::DiagramCall>();
        case 9: return new megamol::core::CallAutoDescription<megamol::protein::SplitMergeCall>();
        case 10: return new megamol::core::CallAutoDescription<megamol::protein::IntSelectionCall>();
        case 11: return new megamol::core::CallAutoDescription<megamol::protein::ResidueSelectionCall>();
        case 12: return new megamol::core::CallAutoDescription<megamol::protein::CrystalStructureDataCall>();
        case 13: return new megamol::core::CallAutoDescription<megamol::protein::VTIDataCall>();
        case 14: return new megamol::core::CallAutoDescription<megamol::protein::VariantMatchDataCall>();
        case 15: return new megamol::core::CallAutoDescription<megamol::protein::VBODataCall>();
        case 16: return new megamol::core::CallAutoDescription<megamol::protein::VTKLegacyDataCallUnstructuredGrid>();
		case 17: return new megamol::core::CallAutoDescription<megamol::protein::CallColor>();
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

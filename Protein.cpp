/*
 * Protein.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "Protein.h"
#include "api/MegaMolCore.std.h"

// jobs
#include "DataWriter.h"

// views
#include "View3DSpaceMouse.h"
#include "View3DMouse.h"

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
#include "BindingSiteDataSource.h"

// data interfaces (calls)
#include "SolPathDataCall.h"
#include "CallVolumeData.h"
#include "MolecularDataCall.h"
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
#include "BindingSiteCall.h"
#include "MoleculeBallifier.h"

// other modules
#include "IntSelection.h"
#include "ResidueSelection.h"

#include "CallAutoDescription.h"
#include "ModuleAutoDescription.h"
#include "vislib/vislibversion.h"

#include "vislib/Log.h"
#include "vislib/Trace.h"
#include "vislib/ThreadSafeStackTrace.h"



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
    int moduleCount = 35;
#ifdef WITH_CUDA
    moduleCount+=9;
#endif // WITH_CUDA
#ifdef WITH_OPENHAPTICS
    moduleCount+=1;
#endif // WITH_OPENHAPTICS
    return moduleCount;
}


/*
 * mmplgModuleDescription
 */
PROTEIN_API void* mmplgModuleDescription(int idx) {
    switch (idx) {
        case 0: return new megamol::core::ModuleAutoDescription<megamol::protein::MoleculeBallifier>();
        case 1: return new megamol::core::ModuleAutoDescription<megamol::protein::SolPathDataSource>();
        case 2: return new megamol::core::ModuleAutoDescription<megamol::protein::SolPathRenderer>();
        case 3: return new megamol::core::ModuleAutoDescription<megamol::protein::ProteinVolumeRenderer>();
        case 4: return new megamol::core::ModuleAutoDescription<megamol::protein::CCP4VolumeData>();
        case 5: return new megamol::core::ModuleAutoDescription<megamol::protein::PDBLoader>();
        case 6: return new megamol::core::ModuleAutoDescription<megamol::protein::SimpleMoleculeRenderer>();
        case 7: return new megamol::core::ModuleAutoDescription<megamol::protein::CoarseGrainDataLoader>();
        case 8: return new megamol::core::ModuleAutoDescription<megamol::protein::SphereRenderer>();
        case 9: return new megamol::core::ModuleAutoDescription<megamol::protein::MoleculeSESRenderer>();
        case 10: return new megamol::core::ModuleAutoDescription<megamol::protein::MoleculeCartoonRenderer>();
        case 11: return new megamol::core::ModuleAutoDescription<megamol::protein::FrodockLoader>();
        case 12: return new megamol::core::ModuleAutoDescription<megamol::protein::VolumeSliceRenderer>();
        case 13: return new megamol::core::ModuleAutoDescription<megamol::protein::Diagram2DRenderer>();
        case 14: return new megamol::core::ModuleAutoDescription<megamol::protein::XYZLoader>();
        case 15: return new megamol::core::ModuleAutoDescription<megamol::protein::ElectrostaticsRenderer>();
        case 16: return new megamol::core::ModuleAutoDescription<megamol::protein::SolventVolumeRenderer>();
        case 17: return new megamol::core::ModuleAutoDescription<megamol::protein::Filter>();
        case 18: return new megamol::core::ModuleAutoDescription<megamol::protein::SolventDataGenerator>();
        case 19: return new megamol::core::ModuleAutoDescription<megamol::protein::View3DSpaceMouse>();
        case 20: return new megamol::core::ModuleAutoDescription<megamol::protein::GROLoader>();
        case 21: return new megamol::core::ModuleAutoDescription<megamol::protein::SSAORendererDeferred>();
        case 22: return new megamol::core::ModuleAutoDescription<megamol::protein::ToonRendererDeferred>();
        case 23: return new megamol::core::ModuleAutoDescription<megamol::protein::DofRendererDeferred>();
        case 24: return new megamol::core::ModuleAutoDescription<megamol::protein::SphereRendererMouse>();
        case 25: return new megamol::core::ModuleAutoDescription<megamol::protein::View3DMouse>();
        case 26: return new megamol::core::ModuleAutoDescription<megamol::protein::GLSLVolumeRenderer>();
        case 27: return new megamol::core::ModuleAutoDescription<megamol::protein::DiagramRenderer>();
        case 28: return new megamol::core::ModuleAutoDescription<megamol::protein::SplitMergeRenderer>();
        case 29: return new megamol::core::ModuleAutoDescription<megamol::protein::IntSelection>();
        case 30: return new megamol::core::ModuleAutoDescription<megamol::protein::ResidueSelection>();
        case 31: return new megamol::core::ModuleAutoDescription<megamol::protein::CrystalStructureDataSource>();
#ifdef WITH_CUDA
        case 32: return new megamol::core::ModuleAutoDescription<megamol::protein::MoleculeCudaSESRenderer>();
        case 33: return new megamol::core::ModuleAutoDescription<megamol::protein::MoleculeCBCudaRenderer>();
        case 34: return new megamol::core::ModuleAutoDescription<megamol::protein::QuickSurfRenderer>();
        case 35: return new megamol::core::ModuleAutoDescription<megamol::protein::QuickSurfRenderer2>();
        case 36: return new megamol::core::ModuleAutoDescription<megamol::protein::QuickSurfMTRenderer>();
        case 37: return new megamol::core::ModuleAutoDescription<megamol::protein::MoleculeVolumeCudaRenderer>();
        case 38: return new megamol::core::ModuleAutoDescription<megamol::protein::VolumeMeshRenderer>();
        case 39: return new megamol::core::ModuleAutoDescription<megamol::protein::DataWriter>();
        case 40: return new megamol::core::ModuleAutoDescription<megamol::protein::CrystalStructureVolumeRenderer>();
        #define CUDA_OFFSET 9
#else
        #define CUDA_OFFSET 0
#endif // WITH_CUDA
#ifdef WITH_OPENHAPTICS
        case 32 + CUDA_OFFSET: return new megamol::core::ModuleAutoDescription<megamol::protein::HapticsMoleculeRenderer>();
        #define HAPTICS_OFFSET 1
#else
        #define HAPTICS_OFFSET 0
#endif // WITH_OPENHAPTICS
        case 32 + CUDA_OFFSET + HAPTICS_OFFSET : return new megamol::core::ModuleAutoDescription<megamol::protein::SequenceRenderer>();
        case 33 + CUDA_OFFSET + HAPTICS_OFFSET : return new megamol::core::ModuleAutoDescription<megamol::protein::BindingSiteDataSource>();
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgCallCount
 */
PROTEIN_API int mmplgCallCount(void) {
    return 15;
}


/*
 * mmplgCallDescription
 */
PROTEIN_API void* mmplgCallDescription(int idx) {
    switch (idx) {
        case 0: return new megamol::core::CallAutoDescription<megamol::protein::SolPathDataCall>();
        case 1: return new megamol::core::CallAutoDescription<megamol::protein::CallVolumeData>();
        case 2: return new megamol::core::CallAutoDescription<megamol::protein::MolecularDataCall>();
        case 3: return new megamol::core::CallAutoDescription<megamol::protein::SphereDataCall>();
        case 4: return new megamol::core::CallAutoDescription<megamol::protein::VolumeSliceCall>();
        case 5: return new megamol::core::CallAutoDescription<megamol::protein::Diagram2DCall>();
        case 6: return new megamol::core::CallAutoDescription<megamol::protein::ParticleDataCall>();
        case 7: return new megamol::core::CallAutoDescription<megamol::protein::ForceDataCall>();
        case 8: return new megamol::core::CallAutoDescription<megamol::protein::CallMouseInput>();
        case 9: return new megamol::core::CallAutoDescription<megamol::protein::DiagramCall>();
        case 10: return new megamol::core::CallAutoDescription<megamol::protein::SplitMergeCall>();
        case 11: return new megamol::core::CallAutoDescription<megamol::protein::IntSelectionCall>();
        case 12: return new megamol::core::CallAutoDescription<megamol::protein::ResidueSelectionCall>();
        case 13: return new megamol::core::CallAutoDescription<megamol::protein::CrystalStructureDataCall>();
        case 14: return new megamol::core::CallAutoDescription<megamol::protein::BindingSiteCall>();
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

/*
 * Protein.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "../include/protein/Protein.h"
#include "mmcore/api/MegaMolCore.std.h"

// views
#include "View3DSpaceMouse.h"
#include "View3DMouse.h"

// jobs
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
#include "ElectrostaticsRenderer.h"
#include "HapticsMoleculeRenderer.h"
#include "SSAORendererDeferred.h"
#include "ToonRendererDeferred.h"
#include "DofRendererDeferred.h"
#include "SphereRendererMouse.h"
#include "GLSLVolumeRenderer.h"
#include "StreamlineRenderer.h"
#include "VariantMatchRenderer.h"
#include "SecPlaneRenderer.h"
#include "UnstructuredGridRenderer.h"
#include "VolumeDirectionRenderer.h"
#include "LayeredIsosurfaceRenderer.h"
#include "CartoonRenderer.h"
#include "CartoonTessellationRenderer.h"

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
#include "CallMouseInput.h"
#include "VariantMatchDataCall.h"
#include "VTKLegacyDataCallUnstructuredGrid.h"


#include "MoleculeBallifier.h"

// other modules (filter etc)
#include "ColorModule.h"
#include "IntSelection.h"
#include "MultiParticleDataFilter.h"
#include "PDBInterpolator.h"

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
    int moduleCount = 52;
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
		case 0 : return new megamol::core::factories::ModuleAutoDescription<megamol::protein::HapticsMoleculeRenderer>();
        #define HAPTICS_OFFSET 1
#else
        #define HAPTICS_OFFSET 0
#endif // WITH_OPENHAPTICS
#ifdef WITH_OPENBABEL
		case CUDA_OFFSET + HAPTICS_OFFSET + 0: return new factories::ModuleAutoDescription<protein::OpenBabelLoader>();
        #define OPENBABAEL_OFFSET 1
#else
        #define OPENBABAEL_OFFSET 0
#endif //WITH_OPENBABEL
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  0 : return new factories::ModuleAutoDescription<protein::SequenceRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  1 : return new factories::ModuleAutoDescription<protein::BindingSiteDataSource>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  2 : return new factories::ModuleAutoDescription<protein::SolPathDataSource>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  3 : return new factories::ModuleAutoDescription<protein::SolPathRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  4 : return new factories::ModuleAutoDescription<protein::ProteinVolumeRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  5 : return new factories::ModuleAutoDescription<protein::CCP4VolumeData>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  6 : return new factories::ModuleAutoDescription<protein::PDBLoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  7 : return new factories::ModuleAutoDescription<protein::SimpleMoleculeRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  8 : return new factories::ModuleAutoDescription<protein::CoarseGrainDataLoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET +  9 : return new factories::ModuleAutoDescription<protein::SphereRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 10 : return new factories::ModuleAutoDescription<protein::MoleculeSESRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 11 : return new factories::ModuleAutoDescription<protein::MoleculeCartoonRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 12 : return new factories::ModuleAutoDescription<protein::FrodockLoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 13 : return new factories::ModuleAutoDescription<protein::VolumeSliceRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 14 : return new factories::ModuleAutoDescription<protein::Diagram2DRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 15 : return new factories::ModuleAutoDescription<protein::XYZLoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 16 : return new factories::ModuleAutoDescription<protein::ElectrostaticsRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 17 : return new factories::ModuleAutoDescription<protein::SolventVolumeRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 18 : return new factories::ModuleAutoDescription<protein::SolventDataGenerator>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 19 : return new factories::ModuleAutoDescription<protein::View3DSpaceMouse>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 20 : return new factories::ModuleAutoDescription<protein::GROLoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 21 : return new factories::ModuleAutoDescription<protein::SSAORendererDeferred>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 22 : return new factories::ModuleAutoDescription<protein::ToonRendererDeferred>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 23 : return new factories::ModuleAutoDescription<protein::DofRendererDeferred>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 24 : return new factories::ModuleAutoDescription<protein::SphereRendererMouse>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 25 : return new factories::ModuleAutoDescription<protein::View3DMouse>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 26 : return new factories::ModuleAutoDescription<protein::GLSLVolumeRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 27 : return new factories::ModuleAutoDescription<protein::DiagramRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 28 : return new factories::ModuleAutoDescription<protein::SplitMergeRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 29 : return new factories::ModuleAutoDescription<protein::IntSelection>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 30 : return new factories::ModuleAutoDescription<protein::CrystalStructureDataSource>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 31 : return new factories::ModuleAutoDescription<protein::VTILoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 32 : return new factories::ModuleAutoDescription<protein::PDBWriter>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 33 : return new factories::ModuleAutoDescription<protein::VTIWriter>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 34 : return new factories::ModuleAutoDescription<protein::VMDDXLoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 35 : return new factories::ModuleAutoDescription<protein::TrajectorySmoothFilter>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 36 : return new factories::ModuleAutoDescription<protein::VariantMatchRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 37 : return new factories::ModuleAutoDescription<protein::MoleculeBallifier>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 38 : return new factories::ModuleAutoDescription<protein::ResidueSelection>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 39 : return new factories::ModuleAutoDescription<protein::SecPlaneRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 40 : return new factories::ModuleAutoDescription<protein::AggregatedDensity>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 41 : return new factories::ModuleAutoDescription<protein::VTKLegacyDataLoaderUnstructuredGrid>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 42 : return new factories::ModuleAutoDescription<protein::UnstructuredGridRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 43 : return new factories::ModuleAutoDescription<protein::MolecularBezierData>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 44 : return new factories::ModuleAutoDescription<protein::MultiParticleDataFilter>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 45 : return new factories::ModuleAutoDescription<protein::VolumeDirectionRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 46 : return new factories::ModuleAutoDescription<protein::LayeredIsosurfaceRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 47 : return new factories::ModuleAutoDescription<protein::MultiPDBLoader>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 48: return new factories::ModuleAutoDescription<protein::ColorModule>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 49: return new factories::ModuleAutoDescription<protein::PDBInterpolator>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 50: return new factories::ModuleAutoDescription<protein::CartoonRenderer>();
        case OPENBABAEL_OFFSET + HAPTICS_OFFSET + 51: return new factories::ModuleAutoDescription<protein::CartoonTessellationRenderer>();
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgCallCount
 */
PROTEIN_API int mmplgCallCount(void) {
    return 11;
}


/*
 * mmplgCallDescription
 */
PROTEIN_API void* mmplgCallDescription(int idx) {
    switch (idx) {
        case 0: return new megamol::core::factories::CallAutoDescription<megamol::protein::SolPathDataCall>();
        case 1: return new megamol::core::factories::CallAutoDescription<megamol::protein::CallProteinVolumeData>();
        case 2: return new megamol::core::factories::CallAutoDescription<megamol::protein::SphereDataCall>();
        case 3: return new megamol::core::factories::CallAutoDescription<megamol::protein::VolumeSliceCall>();
        case 4: return new megamol::core::factories::CallAutoDescription<megamol::protein::Diagram2DCall>();
        case 5: return new megamol::core::factories::CallAutoDescription<megamol::protein::ParticleDataCall>();
        case 6: return new megamol::core::factories::CallAutoDescription<megamol::protein::ForceDataCall>();
        case 7: return new megamol::core::factories::CallAutoDescription<megamol::protein::CallMouseInput>();
        case 8: return new megamol::core::factories::CallAutoDescription<megamol::protein::VariantMatchDataCall>();
        case 9: return new megamol::core::factories::CallAutoDescription<megamol::protein::VTKLegacyDataCallUnstructuredGrid>();
		case 10: return new megamol::core::factories::CallAutoDescription<megamol::protein::CallColor>();
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

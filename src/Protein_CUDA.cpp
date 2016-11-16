/*
 * Protein_CUDA.cpp
 *
 * Copyright (C) 2016 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "../include/protein_cuda/Protein_CUDA.h"
#include "mmcore/api/MegaMolCore.std.h"

// jobs
#include "DataWriter.h"

// 3D renderers
#include "MoleculeCudaSESRenderer.h"
#include "MoleculeCBCudaRenderer.h"
#include "CrystalStructureVolumeRenderer.h"
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
#include "vislib/sys/ThreadSafeStackTrace.h"



/*
 * mmplgPluginAPIVersion
 */
PROTEIN_CUDA_API int mmplgPluginAPIVersion(void) {
    return 100;
}


/*
 * mmplgPluginName
 */
PROTEIN_CUDA_API const char * mmplgPluginName(void) {
    return "Protein_CUDA";
}


/*
 * mmplgPluginDescription
 */
PROTEIN_CUDA_API const char * mmplgPluginDescription(void) {
    return "Plugin for protein rendering using CUDA for accelleration (SFB716 D4)";
}


/*
 * mmplgCoreCompatibilityValue
 */
PROTEIN_CUDA_API const void * mmplgCoreCompatibilityValue(void) {
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
PROTEIN_CUDA_API int mmplgModuleCount(void) {
	int moduleCount = 4;
#ifdef WITH_CUDA
    moduleCount+=18;
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
PROTEIN_CUDA_API void* mmplgModuleDescription(int idx) {
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
		case HAPTICS_OFFSET + 14 : return new factories::ModuleAutoDescription<protein_cuda::QuickSurfRaycaster>();
		case HAPTICS_OFFSET + 15 : return new factories::ModuleAutoDescription<protein_cuda::SecStructFlattener>();
        case HAPTICS_OFFSET + 16 : return new factories::ModuleAutoDescription<protein_cuda::ParticlesToMeshConverter>();
		case HAPTICS_OFFSET + 17 : return new factories::ModuleAutoDescription<protein_cuda::SecStructRenderer2D>();
        #define CUDA_OFFSET 18
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
PROTEIN_CUDA_API int mmplgCallCount(void) {
	return 2;
}


/*
 * mmplgCallDescription
 */
PROTEIN_CUDA_API void* mmplgCallDescription(int idx) {
    switch (idx) {
		case 0: return new megamol::core::factories::CallAutoDescription<megamol::protein_cuda::VBODataCall>();
		case 1: return new megamol::core::factories::CallAutoDescription<megamol::protein_cuda::PlaneDataCall>();
        default: return NULL;
    }
    return NULL;
}


/*
 * mmplgConnectStatics
 */
PROTEIN_CUDA_API bool mmplgConnectStatics(int which, void* value) {
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

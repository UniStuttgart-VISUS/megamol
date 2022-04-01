/**
 * MegaMol
 * Copyright (c) 2016-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

// jobs
#include "DataWriter.h"

// 3D renderers
#include "ComparativeMolSurfaceRenderer.h"
#include "CrystalStructureVolumeRenderer.h"
#include "MoleculeCBCudaRenderer.h"
#include "StreamlineRenderer.h"
#include "SurfacePotentialRendererSlave.h"
#include "VolumeMeshRenderer.h"
#include "QuickSurf.h"

// 2D renderers
#include "SecStructRenderer2D.h"

// data sources
#include "Filter.h"

// data interfaces (calls)
#include "PlaneDataCall.h"
#include "VBODataCall.h"

// other modules (filter etc)
#include "PotentialCalculator.h"
#include "SecStructFlattener.h"
#include "SombreroWarper.h"

#include "vislib/Trace.h"

namespace megamol::protein_cuda {
class ProteinCudaPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ProteinCudaPluginInstance)

public:
    ProteinCudaPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "Protein_CUDA", "Plugin for protein rendering using CUDA for accelleration (SFB716 D4)"){};

    ~ProteinCudaPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::StreamlineRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::MoleculeCBCudaRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::VolumeMeshRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::DataWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::CrystalStructureVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::ComparativeMolSurfaceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SecStructFlattener>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SecStructRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::PotentialCalculator>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::QuickSurf>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SurfacePotentialRendererSlave>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_cuda::SombreroWarper>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::protein_cuda::VBODataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein_cuda::PlaneDataCall>();
    }
};
} // namespace megamol::protein_cuda

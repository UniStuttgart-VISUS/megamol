/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

// 3D renderers
#include "CartoonTessellationRenderer.h"
#include "GLSLVolumeRenderer.h"
#include "MoleculeCartoonRenderer.h"
#include "MoleculeSESRenderer.h"
#include "SimpleMoleculeRenderer.h"
#include "SolPathRenderer.h"
#include "SombreroMeshRenderer.h"
#include "UncertaintyCartoonRenderer.h"
#include "VariantMatchRenderer.h"

// 2D renderers
#include "DiagramRenderer.h"
#include "RamachandranPlot.h"
#include "SequenceRenderer.h"
#include "SplitMergeRenderer.h"
#include "UncertaintySequenceRenderer.h"

// modules depending on GL calls
#include "MSMSCavityFinder.h"
#include "MSMSGenus0Generator.h"
#include "MSMSMeshLoader.h"
#include "TunnelCutter.h"

#include "vislib/Trace.h"

namespace megamol::protein_gl {
class ProteinGLPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(ProteinGLPluginInstance)

public:
    ProteinGLPluginInstance(void)
            : megamol::core::factories::AbstractPluginInstance(
                  "protein_gl", "Plugin for protein rendering (SFB716 D4)"){};

    ~ProteinGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::SequenceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::SolPathRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::SimpleMoleculeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::MoleculeCartoonRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::GLSLVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::DiagramRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::SplitMergeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::RamachandranPlot>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::VariantMatchRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::UncertaintySequenceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::CartoonTessellationRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::SombreroMeshRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::MoleculeSESRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::UncertaintyCartoonRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::MSMSCavityFinder>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::MSMSGenus0Generator>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::MSMSMeshLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein_gl::TunnelCutter>();

        // register calls
    }
};
} // namespace megamol::protein_gl

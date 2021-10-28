/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

// 3D renderers
#include "CartoonTessellationRenderer2000GT.h"
#include "GLSLVolumeRenderer.h"
#include "MoleculeCartoonRenderer.h"
#include "MoleculeSESRenderer.h"
#include "SecPlaneRenderer.h"
#include "SimpleMoleculeRenderer.h"
#include "SolPathRenderer.h"
#include "SolventVolumeRenderer.h"
#include "SombreroMeshRenderer.h"
#include "UnstructuredGridRenderer.h"
#include "VariantMatchRenderer.h"

// 2D renderers
#include "Diagram2DRenderer.h"
#include "DiagramRenderer.h"
#include "SequenceRenderer.h"
#include "SplitMergeRenderer.h"
#include "VolumeSliceRenderer.h"

#include "vislib/Trace.h"

namespace megamol::protein {
class ProteinGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ProteinGLPluginInstance)

public:
    ProteinGLPluginInstance(void)
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "protein_gl", "Plugin for protein rendering (SFB716 D4)"){};

    ~ProteinGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
#ifdef WITH_OPENHAPTICS
        this->module_descriptions.RegisterAutoDescription<megamol::protein::HapticsMoleculeRenderer>();
#endif // WITH_OPENHAPTICS
#ifdef WITH_OPENBABEL
        this->module_descriptions.RegisterAutoDescription<megamol::protein::OpenBabelLoader>();
#endif // WITH_OPENBABEL
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SequenceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolPathRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SimpleMoleculeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MoleculeCartoonRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VolumeSliceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::Diagram2DRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolventVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::GLSLVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::DiagramRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SplitMergeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VariantMatchRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SecPlaneRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::UnstructuredGridRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CartoonTessellationRenderer2000GT>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SombreroMeshRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MoleculeSESRenderer>();

        // register calls

    }
};
} // namespace megamol::protein

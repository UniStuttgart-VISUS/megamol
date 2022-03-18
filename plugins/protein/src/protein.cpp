/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

// jobs
#include "PDBWriter.h"
#include "VTIWriter.h"


// data sources
#include "AggregatedDensity.h"
#include "BindingSiteDataSource.h"
#include "CCP4VolumeData.h"
#include "CaverTunnelResidueLoader.h"
#include "CoarseGrainDataLoader.h"
#include "CrystalStructureDataSource.h"
#include "FrodockLoader.h"
#include "GROLoader.h"
#include "MolecularBezierData.h"
#include "MultiPDBLoader.h"
#include "OpenBabelLoader.h"
#include "PDBLoader.h"
#include "ResidueSelection.h"
#include "SolPathDataSource.h"
#include "SolventHydroBondGenerator.h"
#include "TrajectorySmoothFilter.h"
#include "UncertaintyDataLoader.h"
#include "VMDDXLoader.h"
#include "VTILoader.h"
#include "VTKLegacyDataLoaderUnstructuredGrid.h"

// data interfaces (calls)
#include "protein/Diagram2DCall.h"
#include "protein/ForceDataCall.h"
#include "protein/SolPathDataCall.h"
#include "protein/SphereDataCall.h"
#include "protein/VTKLegacyDataCallUnstructuredGrid.h"
#include "protein/VolumeSliceCall.h"

#include "MoleculeBallifier.h"

// other modules (filter etc)
#include "HydroBondFilter.h"
#include "IntSelection.h"
#include "MolecularNeighborhood.h"
#include "MultiParticleDataFilter.h"
#include "PDBInterpolator.h"
#include "ProteinAligner.h"
#include "ProteinExploder.h"
#include "SolventCounter.h"
#include "TunnelToBFactor.h"
#include "TunnelToParticles.h"

#include "vislib/Trace.h"

namespace megamol::protein {
class ProteinPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ProteinPluginInstance)

public:
    ProteinPluginInstance(void)
            : megamol::core::utility::plugins::AbstractPluginInstance(
                  "protein", "Plugin for protein rendering (SFB716 D4)"){};

    ~ProteinPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
#ifdef WITH_OPENHAPTICS
        this->module_descriptions.RegisterAutoDescription<megamol::protein::HapticsMoleculeRenderer>();
#endif // WITH_OPENHAPTICS
#ifdef WITH_OPENBABEL
        this->module_descriptions.RegisterAutoDescription<megamol::protein::OpenBabelLoader>();
#endif // WITH_OPENBABEL

        this->module_descriptions.RegisterAutoDescription<megamol::protein::BindingSiteDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolPathDataSource>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::CCP4VolumeData>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::PDBLoader>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::CoarseGrainDataLoader>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::FrodockLoader>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolventHydroBondGenerator>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::GROLoader>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::IntSelection>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CrystalStructureDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VTILoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::PDBWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VTIWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VMDDXLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::TrajectorySmoothFilter>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::MoleculeBallifier>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::ResidueSelection>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::AggregatedDensity>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VTKLegacyDataLoaderUnstructuredGrid>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::MolecularBezierData>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MultiParticleDataFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MultiPDBLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::PDBInterpolator>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::ProteinExploder>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MolecularNeighborhood>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::HydroBondFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolventCounter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::ProteinAligner>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CaverTunnelResidueLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::TunnelToBFactor>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::TunnelToParticles>();

        this->module_descriptions.RegisterAutoDescription<megamol::protein::UncertaintyDataLoader>();


        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::protein::SolPathDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::CallProteinVolumeData>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::SphereDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::VolumeSliceCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::Diagram2DCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::ForceDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::VTKLegacyDataCallUnstructuredGrid>();
    }
};
} // namespace megamol::protein

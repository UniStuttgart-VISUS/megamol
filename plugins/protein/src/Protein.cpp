/*
 * Protein.cpp
 *
 * Copyright (C) 2009 by VISUS (Universitaet Stuttgart)
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/PluginRegister.h"

// jobs
#include "PDBWriter.h"
#include "VTIWriter.h"

// 3D renderers
#include "CartoonTessellationRenderer2000GT.h"
#include "GLSLVolumeRenderer.h"
#include "MoleculeCartoonRenderer.h"
#include "SecPlaneRenderer.h"
#include "SimpleMoleculeRenderer.h"
#include "SolPathRenderer.h"
#include "SolventVolumeRenderer.h"
#include "UnstructuredGridRenderer.h"
#include "VariantMatchRenderer.h"
#include "SombreroMeshRenderer.h"
#include "MoleculeSESRenderer.h"

// 2D renderers
#include "Diagram2DRenderer.h"
#include "DiagramRenderer.h"
#include "SequenceRenderer.h"
#include "SplitMergeRenderer.h"
#include "VolumeSliceRenderer.h"

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
#include "VMDDXLoader.h"
#include "VTILoader.h"
#include "VTKLegacyDataLoaderUnstructuredGrid.h"
#include "XYZLoader.h"

// data interfaces (calls)
#include "CallColor.h"
#include "Diagram2DCall.h"
#include "ForceDataCall.h"
#include "ParticleDataCall.h"
#include "SolPathDataCall.h"
#include "SphereDataCall.h"
#include "VTKLegacyDataCallUnstructuredGrid.h"
#include "VolumeSliceCall.h"
#include "mmcore/CallVolumeData.h"
#include "protein/RMSF.h"

#include "MoleculeBallifier.h"

// other modules (filter etc)
#include "ColorModule.h"
#include "HydroBondFilter.h"
#include "IntSelection.h"
#include "MSMSMeshLoader.h"
#include "MSMSGenus0Generator.h"
#include "MolecularNeighborhood.h"
#include "MultiParticleDataFilter.h"
#include "PDBInterpolator.h"
#include "ProteinAligner.h"
#include "ProteinExploder.h"
#include "SolventCounter.h"
#include "MSMSCavityFinder.h"
#include "TunnelCutter.h"

#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "vislib/Trace.h"
#include "mmcore/utility/log/Log.h"

namespace megamol::protein {
/** Implementing the instance class of this plugin */
class plugin_instance : public ::megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : ::megamol::core::utility::plugins::Plugin200Instance(

              /* machine-readable plugin assembly name */
              "Protein",

              /* human-readable plugin description */
              "Plugin for protein rendering (SFB716 D4)"){

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
        this->module_descriptions.RegisterAutoDescription<megamol::protein::HapticsMoleculeRenderer>();
#endif // WITH_OPENHAPTICS
#ifdef WITH_OPENBABEL
        this->module_descriptions.RegisterAutoDescription<megamol::protein::OpenBabelLoader>();
#endif // WITH_OPENBABEL
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SequenceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::BindingSiteDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolPathDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolPathRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CCP4VolumeData>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::PDBLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SimpleMoleculeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CoarseGrainDataLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MoleculeCartoonRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::FrodockLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VolumeSliceRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::Diagram2DRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::XYZLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolventVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolventHydroBondGenerator>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::GROLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::GLSLVolumeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::DiagramRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SplitMergeRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::IntSelection>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CrystalStructureDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VTILoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::PDBWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VTIWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VMDDXLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::TrajectorySmoothFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VariantMatchRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MoleculeBallifier>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::ResidueSelection>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SecPlaneRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::AggregatedDensity>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::VTKLegacyDataLoaderUnstructuredGrid>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::UnstructuredGridRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MolecularBezierData>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MultiParticleDataFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MultiPDBLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::ColorModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::PDBInterpolator>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CartoonTessellationRenderer2000GT>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::ProteinExploder>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MolecularNeighborhood>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::HydroBondFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SolventCounter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MSMSMeshLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MSMSGenus0Generator>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::ProteinAligner>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::CaverTunnelResidueLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MSMSCavityFinder>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::TunnelCutter>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::SombreroMeshRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::protein::MoleculeSESRenderer>();

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::protein::SolPathDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::CallProteinVolumeData>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::SphereDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::VolumeSliceCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::Diagram2DCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::ParticleDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::ForceDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::VTKLegacyDataCallUnstructuredGrid>();
        this->call_descriptions.RegisterAutoDescription<megamol::protein::CallColor>();
    }
};
} // namespace megamol::protein

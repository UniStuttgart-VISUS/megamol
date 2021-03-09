/*
 * mmstd_datatools.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/utility/plugins/PluginRegister.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "table/TableManipulator.h"
#include "CSVFileSequence.h"
#include "DataFileSequence.h"
#include "DataFileSequenceStepper.h"
#include "DataSetTimeRewriteModule.h"
#include "DumpIColorHistogramModule.h"
#include "EnforceSymmetricParticleColorRanges.h"
#include "ErosionField.h"
#include "ForceCubicCBoxModule.h"
#include "IColAdd.h"
#include "IColInverse.h"
#include "IColRangeFix.h"
#include "IColRangeOverride.h"
#include "IColSelectClassify.h"
#include "IColToIdentity.h"
#include "IndexListIndexColor.h"
#include "LocalBoundingBoxExtractor.h"
#include "MPDCListsConcatenate.h"
#include "MPIParticleCollector.h"
#include "MPIVolumeAggregator.h"
#include "MeshTranslateRotateScale.h"
#include "ModColIRange.h"
#include "MultiParticleRelister.h"
#include "NullParticleWriter.h"
#include "OverrideMultiParticleListGlobalColors.h"
#include "OverrideParticleBBox.h"
#include "OverrideParticleGlobals.h"
#include "ParticleBoxFilter.h"
#include "ParticleBoxGeneratorDataSource.h"
#include "ParticleColorChannelSelect.h"
#include "ParticleColorSignThreshold.h"
#include "ParticleColorSignedDistance.h"
#include "ParticleDataSequenceConcatenate.h"
#include "ParticleDensityOpacityModule.h"
#include "ParticleIColFilter.h"
#include "ParticleIColGradientField.h"
#include "ParticleIdentitySort.h"
#include "ParticleListMergeModule.h"
#include "ParticleListSelector.h"
#include "ParticleNeighborhood.h"
#include "ParticleNeighborhoodGraph.h"
#include "ParticleRelaxationModule.h"
#include "ParticleSortFixHack.h"
#include "ParticleThermodyn.h"
#include "ParticleThinner.h"
#include "ParticleTranslateRotateScale.h"
#include "ParticleVelocities.h"
#include "ParticleVisibilityFromVolume.h"
#include "ParticlesToDensity.h"
#include "ParticleInstantiator.h"
#include "RemapIColValues.h"
#include "SphereDataUnifier.h"
#include "StaticMMPLDProvider.h"
#include "SyncedMMPLDProvider.h"
#include "io/MMGDDDataSource.h"
#include "io/MMGDDWriter.h"
#include "io/PLYDataSource.h"
#include "io/PlyWriter.h"
#include "io/STLDataSource.h"
#include "io/TriMeshSTLWriter.h"
#include "io/CPERAWDataSource.h"
#include "mmstd_datatools/GraphDataCall.h"
#include "mmstd_datatools/MultiIndexListDataCall.h"
#include "mmstd_datatools/ParticleFilterMapDataCall.h"
#include "mmstd_datatools/table/TableDataCall.h"
#include "table/CSVDataSource.h"
#include "table/MMFTDataSource.h"
#include "table/MMFTDataWriter.h"
#include "table/TableColumnFilter.h"
#include "table/TableColumnScaler.h"
#include "table/TableFlagFilter.h"
#include "table/TableJoin.h"
#include "table/TableObserverPlane.h"
#include "table/TableSampler.h"
#include "table/TableSelectionTx.h"
#include "table/TableSort.h"
#include "table/TableWhere.h"
#include "table/TableToLines.h"
#include "table/TableToParticles.h"
#include "MPDCGrid.h"
#include "table/TableSplit.h"
#include "CSVWriter.h"
#include "clustering/ParticleIColClustering.h"
#include "AddParticleColors.h"
#include "ColorToDir.h"

namespace megamol::stdplugin::datatools {
/** Implementing the instance class of this plugin */
class plugin_instance : public megamol::core::utility::plugins::Plugin200Instance {
    REGISTERPLUGIN(plugin_instance)
public:
    /** ctor */
    plugin_instance(void)
        : megamol::core::utility::plugins::Plugin200Instance(
              /* machine-readable plugin assembly name */
              "mmstd_datatools",
              /* human-readable plugin description */
              "MegaMol Standard-Plugin containing data manipulation and conversion modules"){
              // here we could perform addition initialization
          };
    /** Dtor */
    virtual ~plugin_instance(void) {
        // here we could perform addition de-initialization
    }
    /** Registers modules and calls */
    virtual void registerClasses(void) {
        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DataSetTimeRewriteModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleListMergeModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DataFileSequenceStepper>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::SphereDataUnifier>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleThinner>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::OverrideParticleGlobals>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleRelaxationModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleListSelector>();
        this->module_descriptions
            .RegisterAutoDescription<megamol::stdplugin::datatools::ParticleDensityOpacityModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ForceCubicCBoxModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DumpIColorHistogramModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DataFileSequence>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::OverrideParticleBBox>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleColorSignThreshold>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleColorSignedDistance>();
        this->module_descriptions
            .RegisterAutoDescription<megamol::stdplugin::datatools::EnforceSymmetricParticleColorRanges>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleSortFixHack>();
        this->module_descriptions
            .RegisterAutoDescription<megamol::stdplugin::datatools::ParticleDataSequenceConcatenate>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleIColFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MultiParticleRelister>();
        this->module_descriptions
            .RegisterAutoDescription<megamol::stdplugin::datatools::OverrideMultiParticleListGlobalColors>();
        this->module_descriptions
            .RegisterAutoDescription<megamol::stdplugin::datatools::ParticleBoxGeneratorDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::CSVDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::TableToParticles>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::TableToLines>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::MMFTDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::MMFTDataWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleColorChannelSelect>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleIColGradientField>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::RemapIColValues>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleNeighborhoodGraph>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IColInverse>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IColAdd>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ModColIRange>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::NullParticleWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IColRangeFix>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IColRangeOverride>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IndexListIndexColor>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IColSelectClassify>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ErosionField>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::MMGDDWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::MMGDDDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableColumnScaler>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableObserverPlane>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableJoin>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableColumnFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableSampler>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableFlagFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableSelectionTx>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableSort>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableWhere>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleVelocities>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleNeighborhood>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleThermodyn>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::PlyWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MPIParticleCollector>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MPIVolumeAggregator>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticlesToDensity>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MPDCListsConcatenate>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::STLDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::TriMeshSTLWriter>();
        this->module_descriptions
            .RegisterAutoDescription<megamol::stdplugin::datatools::ParticleTranslateRotateScale>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::PLYDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MeshTranslateRotateScale>();
        this->module_descriptions
            .RegisterAutoDescription<megamol::stdplugin::datatools::ParticleVisibilityFromVolume>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::CSVFileSequence>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IColToIdentity>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleIdentitySort>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleBoxFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::StaticMMPLDProvider>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::SyncedMMPLDProvider>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableManipulator>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::CPERAWDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::LocalBoundingBoxExtractor>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleInstantiator>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MPDCGrid>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableSplit>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::CSVWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::clustering::ParticleIColClustering>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::AddParticleColors>();
        this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ColorToDir>();

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleFilterMapDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::GraphDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MultiIndexListDataCall>();
    }
};
} // namespace megamol::stdplugin::datatools

/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "AddParticleColors.h"
#include "CSVFileSequence.h"
#include "CSVWriter.h"
#include "ColorToDir.h"
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
#include "MPDCGrid.h"
#include "MPDCListsConcatenate.h"
#include "MPIParticleCollector.h"
#include "MPIVolumeAggregator.h"
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
#include "ParticleIColFilter.h"
#include "ParticleIColGradientField.h"
#include "ParticleIdentitySort.h"
#include "ParticleInstantiator.h"
#include "ParticleListFilter.h"
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
#include "RemapIColValues.h"
#include "SiffCSplineFitter.h"
#include "SphereDataUnifier.h"
#include "StaticMMPLDProvider.h"
#include "SyncedMMPLDProvider.h"
#include "datatools/GraphDataCall.h"
#include "datatools/MultiIndexListDataCall.h"
#include "datatools/ParticleFilterMapDataCall.h"
#include "datatools/clustering/ParticleIColClustering.h"
#include "datatools/table/TableDataCall.h"
#include "io/CPERAWDataSource.h"
#include "io/MMGDDDataSource.h"
#include "io/MMGDDWriter.h"
#include "table/CSVDataSource.h"
#include "table/MMFTDataSource.h"
#include "table/MMFTDataWriter.h"
#include "table/ParticlesToTable.h"
#include "table/TableColumnFilter.h"
#include "table/TableColumnScaler.h"
#include "table/TableFlagFilter.h"
#include "table/TableInspector.h"
#include "table/TableItemSelector.h"
#include "table/TableJoin.h"
#include "table/TableManipulator.h"
#include "table/TableObserverPlane.h"
#include "table/TableSampler.h"
#include "table/TableSelectionTx.h"
#include "table/TableSort.h"
#include "table/TableSplit.h"
#include "table/TableToLines.h"
#include "table/TableToParticles.h"
#include "table/TableWhere.h"

namespace megamol::datatools {
class DatatoolsPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(DatatoolsPluginInstance)

public:
    DatatoolsPluginInstance()
            : megamol::core::factories::AbstractPluginInstance(
                  "datatools", "MegaMol Standard-Plugin containing data manipulation and conversion modules"){};

    ~DatatoolsPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::DataSetTimeRewriteModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::DataFileSequenceStepper>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::SphereDataUnifier>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleThinner>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::OverrideParticleGlobals>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleRelaxationModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleListSelector>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ForceCubicCBoxModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::DumpIColorHistogramModule>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::DataFileSequence>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::OverrideParticleBBox>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleColorSignThreshold>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleColorSignedDistance>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::EnforceSymmetricParticleColorRanges>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleSortFixHack>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleDataSequenceConcatenate>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleIColFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::MultiParticleRelister>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::OverrideMultiParticleListGlobalColors>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleBoxGeneratorDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::CSVDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::TableToParticles>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::TableToLines>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::MMFTDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::MMFTDataWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleColorChannelSelect>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleIColGradientField>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::RemapIColValues>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleNeighborhoodGraph>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::IColInverse>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::IColAdd>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ModColIRange>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::NullParticleWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::IColRangeFix>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::IColRangeOverride>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::IndexListIndexColor>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::IColSelectClassify>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ErosionField>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::io::MMGDDWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::io::MMGDDDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableColumnScaler>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableObserverPlane>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableJoin>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableColumnFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableSampler>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableSort>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableWhere>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableFlagFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableSelectionTx>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleVelocities>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleNeighborhood>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleThermodyn>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::MPIParticleCollector>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::MPIVolumeAggregator>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticlesToDensity>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::MPDCListsConcatenate>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleTranslateRotateScale>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleVisibilityFromVolume>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::CSVFileSequence>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::IColToIdentity>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleIdentitySort>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleBoxFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::StaticMMPLDProvider>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::SyncedMMPLDProvider>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableManipulator>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::io::CPERAWDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::LocalBoundingBoxExtractor>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleInstantiator>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::MPDCGrid>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::table::TableSplit>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::CSVWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::clustering::ParticleIColClustering>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::AddParticleColors>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ColorToDir>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticlesToTable>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::TableInspector>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::ParticleListFilter>();
        this->module_descriptions.RegisterAutoDescription<megamol::datatools::SiffCSplineFitter>();
        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::datatools::table::TableDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::datatools::ParticleFilterMapDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::datatools::GraphDataCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::datatools::MultiIndexListDataCall>();
    }
};
} // namespace megamol::datatools

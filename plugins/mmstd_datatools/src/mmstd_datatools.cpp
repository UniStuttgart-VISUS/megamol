/*
 * mmstd_datatools.cpp
 *
 * Copyright (C) 2009-2015 by MegaMol Team
 * Alle Rechte vorbehalten.
 */

#include "stdafx.h"
#include "mmstd_datatools/mmstd_datatools.h"

#include "mmcore/api/MegaMolCore.std.h"
#include "mmcore/utility/plugins/Plugin200Instance.h"
#include "mmcore/versioninfo.h"
#include "vislib/vislibversion.h"

#include "DataSetTimeRewriteModule.h"
#include "ParticleListMergeModule.h"
#include "DataFileSequenceStepper.h"
#include "SphereDataUnifier.h"
#include "ParticleThinner.h"
#include "OverrideParticleGlobals.h"
#include "ParticleRelaxationModule.h"
#include "ParticleListSelector.h"
#include "ParticleDensityOpacityModule.h"
#include "ForceCubicCBoxModule.h"
#include "DumpIColorHistogramModule.h"
#include "DataFileSequence.h"
#include "OverrideParticleBBox.h"
#include "ParticleColorSignThreshold.h"
#include "ParticleColorSignedDistance.h"
#include "EnforceSymmetricParticleColorRanges.h"
#include "ParticleSortFixHack.h"
#include "ParticleDataSequenceConcatenate.h"
#include "ParticleIColFilter.h"
#include "MultiParticleRelister.h"
#include "OverrideMultiParticleListGlobalColors.h"
#include "ParticleBoxGeneratorDataSource.h"
#include "mmstd_datatools/table/TableDataCall.h"
#include "table/CSVDataSource.h"
#include "table/TableToParticles.h"
#include "table/MMFTDataSource.h"
#include "table/MMFTDataWriter.h"
#include "ParticleColorChannelSelect.h"
#include "ParticleIColGradientField.h"
#include "mmstd_datatools/ParticleFilterMapDataCall.h"
#include "RemapIColValues.h"
#include "mmstd_datatools/GraphDataCall.h"
#include "ParticleNeighborhoodGraph.h"
#include "IColInverse.h"
#include "IColAdd.h"
#include "ModColIRange.h"
#include "NullParticleWriter.h"
#include "IColRangeFix.h"
#include "IColRangeOverride.h"
#include "mmstd_datatools/MultiIndexListDataCall.h"
#include "IndexListIndexColor.h"
#include "IColSelectClassify.h"
#include "ErosionField.h"
#include "io/MMGDDWriter.h"
#include "io/MMGDDDataSource.h"
#include "io/PlyWriter.h"
#include "table/TableColumnScaler.h"
#include "table/TableObserverPlane.h"
#include "table/TableJoin.h"
#include "table/TableColumnFilter.h"
#include "ParticleVelocities.h"
#include "ParticleNeighborhood.h"
#include "ParticleThermodyn.h"
#include "io/PLYDataSource.h"
#include "table/TableToLines.h"
#include "MPIParticleCollector.h"
#include "MPIVolumeAggregator.h"
#include "ParticlesToDensity.h"
#include "MPDCListsConcatenate.h"
#include "io/STLDataSource.h"
#include "io/TriMeshSTLWriter.h"
#include "ParticleTranslateRotateScale.h"
#include "MeshTranslateRotateScale.h"
#include "ParticleVisibilityFromVolume.h"
#include "CSVFileSequence.h"
#include "IColToIdentity.h"
#include "ParticleIdentitySort.h"
#include "ParticleBoxFilter.h"
#include "StaticMMPLDProvider.h"
#include "SyncedMMPLDProvider.h"

/*
 * mmplgPluginAPIVersion
 */
MMSTD_DATATOOLS_API int mmplgPluginAPIVersion(void) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgPluginAPIVersion
}


/*
 * mmplgGetPluginCompatibilityInfo
 */
MMSTD_DATATOOLS_API
megamol::core::utility::plugins::PluginCompatibilityInfo *
mmplgGetPluginCompatibilityInfo(
        megamol::core::utility::plugins::ErrorCallback onError) {
    // compatibility information with core and vislib
    using megamol::core::utility::plugins::PluginCompatibilityInfo;
    using megamol::core::utility::plugins::LibraryVersionInfo;

    PluginCompatibilityInfo *ci = new PluginCompatibilityInfo;
    ci->libs_cnt = 2;
    ci->libs = new LibraryVersionInfo[2];

    SetLibraryVersionInfo(ci->libs[0], "MegaMolCore",
        MEGAMOL_CORE_MAJOR_VER, MEGAMOL_CORE_MINOR_VER, MEGAMOL_CORE_COMP_REV, 0
#if defined(DEBUG) || defined(_DEBUG)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(MEGAMOL_CORE_DIRTY) && (MEGAMOL_CORE_DIRTY != 0)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
        );

    SetLibraryVersionInfo(ci->libs[1], "vislib",
        vislib::VISLIB_VERSION_MAJOR, vislib::VISLIB_VERSION_MINOR, vislib::VISLIB_VERSION_REVISION, 0
#if defined(DEBUG) || defined(_DEBUG)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DEBUG_BUILD
#endif
#if defined(VISLIB_DIRTY_BUILD) && (VISLIB_DIRTY_BUILD != 0)
        | MEGAMOLCORE_PLUGIN200UTIL_FLAGS_DIRTY_BUILD
#endif
        );

    return ci;
}


/*
 * mmplgReleasePluginCompatibilityInfo
 */
MMSTD_DATATOOLS_API void mmplgReleasePluginCompatibilityInfo(
        megamol::core::utility::plugins::PluginCompatibilityInfo* ci) {
    // release compatiblity data on the correct heap
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginCompatibilityInfo(ci)
}


/* anonymous namespace hides this type from any other object files */
namespace {
    /** Implementing the instance class of this plugin */
    class plugin_instance : public megamol::core::utility::plugins::Plugin200Instance {
    public:
        /** ctor */
        plugin_instance(void)
            : megamol::core::utility::plugins::Plugin200Instance(
                /* machine-readable plugin assembly name */
                "mmstd_datatools",
                /* human-readable plugin description */
                "MegaMol Standard-Plugin containing data manipulation and conversion modules") {
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
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleDensityOpacityModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ForceCubicCBoxModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DumpIColorHistogramModule>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::DataFileSequence>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::OverrideParticleBBox>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleColorSignThreshold>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleColorSignedDistance>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::EnforceSymmetricParticleColorRanges>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleSortFixHack>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleDataSequenceConcatenate>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleIColFilter>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MultiParticleRelister>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::OverrideMultiParticleListGlobalColors >();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleBoxGeneratorDataSource>();
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
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleTranslateRotateScale>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::io::PLYDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MeshTranslateRotateScale>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleVisibilityFromVolume>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::CSVFileSequence>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::IColToIdentity>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleIdentitySort>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleBoxFilter>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::StaticMMPLDProvider>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::SyncedMMPLDProvider>();

            // register calls here:
            this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::table::TableDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::ParticleFilterMapDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::GraphDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::datatools::MultiIndexListDataCall>();
        }
        MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_plugininstance_connectStatics
    };
}


/*
 * mmplgGetPluginInstance
 */
MMSTD_DATATOOLS_API
megamol::core::utility::plugins::AbstractPluginInstance*
mmplgGetPluginInstance(
        megamol::core::utility::plugins::ErrorCallback onError) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgGetPluginInstance(plugin_instance, onError)
}


/*
 * mmplgReleasePluginInstance
 */
MMSTD_DATATOOLS_API void mmplgReleasePluginInstance(
        megamol::core::utility::plugins::AbstractPluginInstance* pi) {
    MEGAMOLCORE_PLUGIN200UTIL_IMPLEMENT_mmplgReleasePluginInstance(pi)
}

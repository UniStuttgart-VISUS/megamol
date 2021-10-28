/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "mmstd_moldyn/BrickStatsCall.h"

#include "io/BrickStatsDataSource.h"
#include "io/IMDAtomDataSource.h"
#include "io/MMPGDDataSource.h"
#include "io/MMPGDWriter.h"
#include "io/MMSPDDataSource.h"
#include "io/SIFFDataSource.h"
#include "io/SIFFWriter.h"
#include "io/TclMolSelectionLoader.h"
#include "io/VIMDataSource.h"
#include "io/VTFDataSource.h"
#include "io/VTFResDataSource.h"
#include "io/VisIttDataSource.h"
#include "io/XYZLoader.h"
#include "io/MMPLDDataSource.h"
#include "io/MMPLDWriter.h"
#include "io/TestSpheresDataSource.h"

#include "misc/ParticleWorker.h"

#include "DataGridder.h"
#include "mmstd_moldyn/ParticleGridDataCall.h"

namespace megamol::stdplugin::moldyn {
    class MoldynPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(MoldynPluginInstance)

    public:
        MoldynPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance(
                      "mmstd_moldyn", "MegaMol Plugins for Molecular Dynamics Data Visualization"){};

        ~MoldynPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::IMDAtomDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::MMSPDDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::SIFFDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::SIFFWriter>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::VIMDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::VisIttDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::VTFDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::VTFResDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::misc::ParticleWorker>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::XYZLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::TclMolSelectionLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::BrickStatsDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::DataGridder>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::MMPGDDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::MMPGDWriter>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::MMPLDDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::MMPLDWriter>();
            this->module_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::io::TestSpheresDataSource>();

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::stdplugin::moldyn::BrickStatsCall>();
            this->call_descriptions
                .RegisterAutoDescription<megamol::stdplugin::moldyn::ParticleGridDataCall>();
        }
    };
} // namespace megamol::stdplugin::moldyn

/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "moldyn/BrickStatsCall.h"

#include "io/BrickStatsDataSource.h"
#include "io/IMDAtomDataSource.h"
#include "io/MMPGDDataSource.h"
#include "io/MMPGDWriter.h"
#include "io/MMPLDDataSource.h"
#include "io/MMPLDWriter.h"
#include "io/MMSPDDataSource.h"
#include "io/SIFFDataSource.h"
#include "io/SIFFWriter.h"
#include "io/TclMolSelectionLoader.h"
#include "io/TestSpheresDataSource.h"
#include "io/VIMDataSource.h"
#include "io/VTFDataSource.h"
#include "io/VTFResDataSource.h"
#include "io/VisIttDataSource.h"
#include "io/XYZLoader.h"

#include "DataGridder.h"
#include "moldyn/ParticleGridDataCall.h"

namespace megamol::moldyn {
class MoldynPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(MoldynPluginInstance)

public:
    MoldynPluginInstance()
            : megamol::core::factories::AbstractPluginInstance(
                  "moldyn", "MegaMol Plugins for Molecular Dynamics Data Visualization"){};

    ~MoldynPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::IMDAtomDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::MMSPDDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::SIFFDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::SIFFWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::VIMDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::VisIttDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::VTFDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::VTFResDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::XYZLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::TclMolSelectionLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::BrickStatsDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::DataGridder>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::MMPGDDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::MMPGDWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::MMPLDDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::MMPLDWriter>();
        this->module_descriptions.RegisterAutoDescription<megamol::moldyn::io::TestSpheresDataSource>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::moldyn::BrickStatsCall>();
        this->call_descriptions.RegisterAutoDescription<megamol::moldyn::ParticleGridDataCall>();
    }
};
} // namespace megamol::moldyn

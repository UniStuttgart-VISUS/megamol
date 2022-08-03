/**
 * MegaMol
 * Copyright (c) 2019, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "ArchVisMSMDataSource.h"
#include "CreateFEMModel.h"
#include "CreateMSM.h"
#include "FEMLoader.h"
#include "FEMModel.h"
#include "FEMRenderer.h"
#include "MSMConvexHullRenderer.h"
#include "MSMRenderer.h"

namespace megamol::archvis_gl {
class ArchvisPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ArchvisPluginInstance)

public:
    ArchvisPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("archvis", "The archvis plugin."){};

    ~ArchvisPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::archvis_gl::ArchVisMSMDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::archvis_gl::CreateFEMModel>();
        this->module_descriptions.RegisterAutoDescription<megamol::archvis_gl::CreateMSM>();
        this->module_descriptions.RegisterAutoDescription<megamol::archvis_gl::FEMRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::archvis_gl::FEMLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::archvis_gl::MSMConvexHullRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::archvis_gl::MSMRenderer>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::archvis_gl::CallFEMModel>();
        this->call_descriptions.RegisterAutoDescription<megamol::archvis_gl::CallScaleModel>();
    }
};
} // namespace megamol::archvis_gl

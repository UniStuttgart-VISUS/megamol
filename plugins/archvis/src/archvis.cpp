/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

// TODO: Vislib must die!!!
// Vislib includes Windows.h. This crashes when somebody else (i.e. zmq) is using Winsock2.h, but the vislib include
// is first without defining WIN32_LEAN_AND_MEAN. This define is the only thing we need from stdafx.h, include could be
// removed otherwise.
#include "stdafx.h"

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "ArchVisMSMDataSource.h"
#include "CreateFEMModel.h"
#include "CreateMSM.h"
#include "FEMDataCall.h"
#include "FEMLoader.h"
#include "FEMMaterialDataSource.h"
#include "FEMMeshDataSource.h"
#include "FEMModel.h"
#include "FEMRenderTaskDataSource.h"
#include "MSMConvexHullMeshDataSource.h"
#include "MSMDataCall.h"
#include "MSMRenderTaskDataSource.h"

namespace megamol::archvis {
    class ArchvisPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
        REGISTERPLUGIN(ArchvisPluginInstance)

    public:
        ArchvisPluginInstance()
                : megamol::core::utility::plugins::AbstractPluginInstance("archvis", "The archvis plugin."){};

        ~ArchvisPluginInstance() override = default;

        // Registers modules and calls
        void registerClasses() override {

            // register modules
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::ArchVisMSMDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::CreateFEMModel>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::CreateMSM>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMMeshDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMMaterialDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMRenderTaskDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::FEMLoader>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::MSMConvexHullDataSource>();
            this->module_descriptions.RegisterAutoDescription<megamol::archvis::MSMRenderTaskDataSource>();

            // register calls
            this->call_descriptions.RegisterAutoDescription<megamol::archvis::FEMDataCall>();
            this->call_descriptions.RegisterAutoDescription<megamol::archvis::MSMDataCall>();
        }
    };
} // namespace megamol::archvis

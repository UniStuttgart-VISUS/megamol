/**
 * MegaMol
 * Copyright (c) 2022, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "cluster/CallRegisterAtController.h"
#include "cluster/ClusterController.h"
#include "cluster/SyncDataSourcesCall.h"
#include "cluster/mpi/MpiCall.h"
#include "cluster/mpi/MpiProvider.h"

namespace megamol::cluster {
class PluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(PluginInstance)
public:
    PluginInstance() : megamol::core::utility::plugins::AbstractPluginInstance("cluster", "Cluster plugin."){};

    ~PluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<core::cluster::ClusterController>();
        this->module_descriptions.RegisterAutoDescription<core::cluster::mpi::MpiProvider>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<core::cluster::CallRegisterAtController>();
        this->call_descriptions.RegisterAutoDescription<core::cluster::mpi::MpiCall>();
        this->call_descriptions.RegisterAutoDescription<core::cluster::SyncDataSourcesCall>();
    }
};
} // namespace megamol::cluster

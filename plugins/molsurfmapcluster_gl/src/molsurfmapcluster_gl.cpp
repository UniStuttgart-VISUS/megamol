/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/factories/AbstractPluginInstance.h"
#include "mmcore/factories/PluginRegister.h"

#include "CallClustering_2.h"
#include "ClusterGraphRenderer.h"
#include "Clustering_2.h"
#include "ProteinViewRenderer.h"

namespace megamol::molsurfmapcluster_gl {
class ProteinCallsPluginInstance : public megamol::core::factories::AbstractPluginInstance {
    REGISTERPLUGIN(ProteinCallsPluginInstance)

public:
    ProteinCallsPluginInstance()
            : megamol::core::factories::AbstractPluginInstance("MolSurfMapCluster",
                  "Plugin containing all necessary modules to produce a clustering of Molecular Surface Maps"){};

    ~ProteinCallsPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::Clustering_2>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::ClusterGraphRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::ProteinViewRenderer>();

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::CallClustering_2>();
    }
};
} // namespace megamol::molsurfmapcluster_gl

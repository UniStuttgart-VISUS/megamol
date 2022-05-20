/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "CallCluster.h"
#include "CallClusterPosition.h"
#include "CallClusteringLoader.h"
#include "CallClustering_2.h"
#include "CallPNGPics.h"
#include "ClusterGraphRenderer.h"
#include "ClusterHierarchieRenderer.h"
#include "ClusterMapRenderer.h"
#include "ClusterRenderer.h"
#include "Clustering.h"
#include "ClusteringLoader.h"
#include "Clustering_2.h"
#include "PNGPicLoader.h"
#include "ProteinViewRenderer.h"

namespace megamol::molsurfmapcluster {
class ProteinCallsPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(ProteinCallsPluginInstance)

public:
    ProteinCallsPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("MolSurfMapCluster",
                  "Plugin containing all necessary modules to produce a clustering of Molecular Surface Maps"){};

    ~ProteinCallsPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules here:
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::PNGPicLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::Clustering>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::ClusterRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::ClusterHierarchieRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::ClusteringLoader>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::Clustering_2>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::ClusterMapRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::ClusterGraphRenderer>();
        this->module_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::ProteinViewRenderer>();

        // register calls here:
        this->call_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::CallPNGPics>();
        this->call_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::CallClustering>();
        this->call_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::CallClusterPosition>();
        this->call_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster::CallClusteringLoader>();
        this->call_descriptions.RegisterAutoDescription<megamol::molsurfmapcluster_gl::CallClustering_2>();
    }
};
} // namespace megamol::molsurfmapcluster

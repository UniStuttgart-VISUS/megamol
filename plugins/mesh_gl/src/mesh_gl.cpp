/**
 * MegaMol
 * Copyright (c) 2009-2021, MegaMol Dev Team
 * All rights reserved.
 */

#include "mmcore/utility/plugins/AbstractPluginInstance.h"
#include "mmcore/utility/plugins/PluginRegister.h"

#include "3DUIRenderTaskDataSource.h"
#include "GPUMeshes.h"
#include "MeshViewerRenderTasks.h"
#include "Render3DUI.h"
#include "RenderMDIMesh.h"
#include "SimpleGPUMtlDataSource.h"
#include "gltf/glTFRenderTasksDataSource.h"
#include "mesh_gl/MeshCalls_gl.h"

namespace megamol::mesh {
class MeshGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MeshGLPluginInstance)

public:
    MeshGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("mesh_gl", "Plugin for rendering meshes."){};

    ~MeshGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::RenderMDIMesh>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GlTFRenderTasksDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::GPUMeshes>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::SimpleGPUMtlDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::MeshViewerRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::Render3DUI>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh::ThreeDimensionalUIRenderTaskDataSource>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMeshData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPUMaterialData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh::CallGPURenderTaskData>();
    }
};
} // namespace megamol::mesh

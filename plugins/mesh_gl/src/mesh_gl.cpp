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
#include "STLDataSource.h"
#include "SimpleGPUMtlDataSource.h"
#include "TriangleMeshRenderer2D.h"
#include "TriangleMeshRenderer3D.h"
#include "gltf/glTFRenderTasksDataSource.h"
#include "mesh_gl/MeshCalls_gl.h"

namespace megamol::mesh_gl {
class MeshGLPluginInstance : public megamol::core::utility::plugins::AbstractPluginInstance {
    REGISTERPLUGIN(MeshGLPluginInstance)

public:
    MeshGLPluginInstance()
            : megamol::core::utility::plugins::AbstractPluginInstance("mesh_gl", "Plugin for rendering meshes."){};

    ~MeshGLPluginInstance() override = default;

    // Registers modules and calls
    void registerClasses() override {

        // register modules
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::RenderMDIMesh>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::GlTFRenderTasksDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::GPUMeshes>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::SimpleGPUMtlDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::MeshViewerRenderTasks>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::Render3DUI>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::STLDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::ThreeDimensionalUIRenderTaskDataSource>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::TriangleMeshRenderer2D>();
        this->module_descriptions.RegisterAutoDescription<megamol::mesh_gl::TriangleMeshRenderer3D>();

        // register calls
        this->call_descriptions.RegisterAutoDescription<megamol::mesh_gl::CallGPUMeshData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh_gl::CallGPUMaterialData>();
        this->call_descriptions.RegisterAutoDescription<megamol::mesh_gl::CallGPURenderTaskData>();
    }
};
} // namespace megamol::mesh_gl
